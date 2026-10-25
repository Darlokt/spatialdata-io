"""Channel selection, order, names, consistency, canvas, and nuclei split of one stack."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image import TiffMetadata
from spatialdata_io.readers._utils.paths import ArtifactDirectory
from spatialdata_io.readers.macsima._channels import (
    NO_PADDING,
    ChannelOptions,
    PlannedChannel,
    make_channel_options,
    plan_stack,
)
from spatialdata_io.readers.macsima._layout import StackFiles
from spatialdata_io.readers.macsima._metadata import ChannelMetadata, ChannelSource

if TYPE_CHECKING:
    from spatialdata_io.readers.macsima._schemas import ImageType

_OPTIONS = ChannelOptions(
    exclude_cycles=frozenset(),
    include_cycle_in_channel_names=False,
    nuclei_channel_name="DAPI",
    split_threshold_nuclei_channel=2,
)
_STACK = StackFiles(name="run", directory=ArtifactDirectory(Path("/data/run")), files=())


def _source(  # noqa: PLR0913, RUF100 - Each keyword varies one planned property.
    name: str,
    cycle: int,
    *,
    image_type: ImageType = "stain",
    path: str | None = None,
    shape: tuple[int, int] = (4, 5),
    offset: tuple[int, int] = (0, 0),
    well: str | None = "D01",
    roi: int | None = 1,
) -> ChannelSource:
    metadata = ChannelMetadata(
        name=name,
        cycle=cycle,
        image_type=image_type,
        well=well,
        roi=roi,
        fluorophore="PE",
        clone=None,
        exposure_ms=10.0,
    )
    return ChannelSource(
        tiff=TiffMetadata(path=Path(path or f"/data/run/{name}_{cycle}.tif"), series=(), ome=None),
        software_version="1.1.1",
        major=1,
        metadata=metadata,
        pixel_size_um=0.17,
        offset_yx=offset,
        shape_yx=shape,
        dtype=np.dtype(np.uint16),
    )


def _names(channels: tuple[PlannedChannel, ...] | None) -> list[str]:
    assert channels is not None
    return [channel.name for channel in channels]


class TestMakeChannelOptions:
    """Validate caller choices once."""

    def test_owns_cycles(self) -> None:
        cycles = [3, 1]

        options = make_channel_options(
            exclude_cycles=cycles,
            include_cycle_in_channel_names=True,
            nuclei_channel_name="Hoechst",
            split_threshold_nuclei_channel=None,
        )

        assert options == ChannelOptions(
            exclude_cycles=frozenset({1, 3}),
            include_cycle_in_channel_names=True,
            nuclei_channel_name="Hoechst",
            split_threshold_nuclei_channel=None,
        )

    @pytest.mark.parametrize(
        ("overrides", "error", "match"),
        [
            ({"exclude_cycles": "12"}, TypeError, "sequence of cycle numbers"),
            ({"exclude_cycles": (1.0,)}, TypeError, "only integers"),
            ({"exclude_cycles": (True,)}, TypeError, "only integers"),
            ({"include_cycle_in_channel_names": 1}, TypeError, "must be a boolean"),
            ({"nuclei_channel_name": 1}, TypeError, "must be a string"),
            ({"nuclei_channel_name": ""}, ValueError, "must not be empty"),
            ({"split_threshold_nuclei_channel": "2"}, TypeError, "None or an integer"),
            ({"split_threshold_nuclei_channel": True}, TypeError, "None or an integer"),
            ({"split_threshold_nuclei_channel": -1}, ValueError, "must not be negative"),
        ],
    )
    def test_rejects(self, overrides: dict[str, object], error: type[Exception], match: str) -> None:
        arguments: dict[str, object] = {
            "exclude_cycles": (),
            "include_cycle_in_channel_names": False,
            "nuclei_channel_name": "DAPI",
            "split_threshold_nuclei_channel": 2,
            **overrides,
        }
        with pytest.raises(error, match=match):
            make_channel_options(**arguments)  # pyrefly: ignore[bad-argument-type] - Invalid runtime inputs.


class TestPlanStack:
    """Plan one stack from inspected channel sources."""

    def test_orders_by_cycle_then_name_then_path(self) -> None:
        sources = [
            _source("Zeta", 2),
            _source("CD3", 1, path="/data/b.tif"),
            _source("Alpha", 2),
            _source("CD3", 1, path="/data/a.tif"),
        ]

        plan = plan_stack(_STACK, sources, _OPTIONS)

        assert _names(plan.channels) == ["CD3 (1)", "CD3 (2)", "Alpha", "Zeta"]
        assert [channel.source.tiff.path.name for channel in plan.channels[:2]] == ["a.tif", "b.tif"]
        assert plan.nuclei is None
        assert (plan.name, plan.pixel_size_um) == ("run", 0.17)

    def test_cycle_prefixed_names_are_numbered_when_repeated(self) -> None:
        sources = [_source("CD3", 1, path="/data/a.tif"), _source("CD3", 1, path="/data/b.tif"), _source("CD3", 2)]

        plan = plan_stack(_STACK, sources, replace(_OPTIONS, include_cycle_in_channel_names=True))

        assert _names(plan.channels) == ["R1 CD3 (1)", "R1 CD3 (2)", "R2 CD3"]

    def test_rejects_generated_name_collisions(self) -> None:
        sources = [_source("CD3", 1), _source("CD3", 2), _source("CD3 (1)", 3)]

        with pytest.raises(ReaderFormatError, match="unique names after numbering") as caught:
            plan_stack(_STACK, sources, _OPTIONS)
        assert caught.value.path == _STACK.directory

    def test_excludes_cycles(self) -> None:
        sources = [_source("CD3", 1), _source("CD4", 2), _source("CD8", 3)]

        plan = plan_stack(_STACK, sources, replace(_OPTIONS, exclude_cycles=frozenset({2, 9})))

        assert _names(plan.channels) == ["CD3", "CD8"]

    def test_rejects_excluding_every_channel(self) -> None:
        with pytest.raises(ValueError, match="removes every channel of stack 'run' in '/data/run'"):
            plan_stack(_STACK, [_source("CD3", 1)], replace(_OPTIONS, exclude_cycles=frozenset({1})))

    @pytest.mark.parametrize(
        ("other", "field"),
        [
            (replace(_source("CD4", 2), dtype=np.dtype(np.uint8)), "dtype"),
            (replace(_source("CD4", 2), pixel_size_um=0.34), r"pixel size \(µm\)"),
            (replace(_source("CD4", 2), major=0), "MACSiQView major version"),
            (_source("CD4", 2, well="D02"), "well"),
            (_source("CD4", 2, roi=2), "ROI"),
        ],
    )
    def test_rejects_inconsistent_channels(self, other: ChannelSource, field: str) -> None:
        with pytest.raises(ReaderFormatError, match=f"one {field} per stack") as caught:
            plan_stack(_STACK, [_source("CD3", 1), other], _OPTIONS)
        assert caught.value.path == _STACK.directory

    def test_missing_well_and_roi_do_not_conflict(self) -> None:
        plan = plan_stack(_STACK, [_source("CD3", 1), _source("CD4", 2, well=None, roi=None)], _OPTIONS)

        assert len(plan.channels) == 2

    def test_unequal_shapes_pad_right_and_bottom_with_a_warning(self) -> None:
        sources = [_source("CD3", 1, shape=(10, 12)), _source("CD4", 2, shape=(8, 14))]

        with pytest.warns(UserWarning, match=r"stack 'run' to the common canvas \(10, 14\)"):
            plan = plan_stack(_STACK, sources, _OPTIONS)

        assert [channel.padding for channel in plan.channels] == [((0, 0), (0, 2)), ((0, 2), (0, 0))]

    def test_offsets_place_equal_shapes_on_a_shared_canvas(self) -> None:
        # Equal shapes with different positions must still be aligned.
        sources = [_source("CD3", 1, offset=(5, 7)), _source("CD4", 2, offset=(2, 9))]

        with pytest.warns(UserWarning, match=r"common canvas \(7, 7\)"):
            plan = plan_stack(_STACK, sources, _OPTIONS)

        assert [channel.padding for channel in plan.channels] == [((3, 0), (0, 2)), ((0, 3), (2, 0))]

    def test_equal_offsets_need_no_padding(self, recwarn: pytest.WarningsRecorder) -> None:
        plan = plan_stack(_STACK, [_source("CD3", 1, offset=(4, 4)), _source("CD4", 2, offset=(4, 4))], _OPTIONS)

        assert {channel.padding for channel in plan.channels} == {NO_PADDING}
        assert not recwarn.list


class TestPlanStackNuclei:
    """Split exact-name stain nuclei channels above the threshold."""

    _SOURCES = (
        _source("CD3", 1),
        _source("DAPI", 1),
        _source("DAPI", 3),
        _source("DAPI", 5),
        _source("DAPI_background", 5, image_type="bleach"),
        _source("DAPI2", 6),
    )

    def test_splits_and_keeps_first_and_last_in_the_main_stack(self) -> None:
        plan = plan_stack(_STACK, self._SOURCES, _OPTIONS)

        assert _names(plan.channels) == ["CD3", "DAPI (1)", "DAPI (3)", "DAPI_background", "DAPI2"]
        assert _names(plan.nuclei) == ["DAPI (1)", "DAPI (2)", "DAPI (3)"]

    def test_threshold_equal_to_count_does_not_split(self) -> None:
        plan = plan_stack(_STACK, self._SOURCES, replace(_OPTIONS, split_threshold_nuclei_channel=3))

        assert plan.nuclei is None
        assert len(plan.channels) == len(self._SOURCES)

    def test_none_never_splits(self) -> None:
        assert plan_stack(_STACK, self._SOURCES, replace(_OPTIONS, split_threshold_nuclei_channel=None)).nuclei is None

    def test_zero_threshold_splits_a_single_nuclei_channel(self) -> None:
        plan = plan_stack(_STACK, self._SOURCES[:2], replace(_OPTIONS, split_threshold_nuclei_channel=0))

        assert _names(plan.channels) == ["CD3", "DAPI"]
        assert _names(plan.nuclei) == ["DAPI"]

    def test_other_nuclear_stain_name(self) -> None:
        sources = [_source("Hoechst", cycle) for cycle in (1, 2, 3)]

        plan = plan_stack(_STACK, sources, replace(_OPTIONS, nuclei_channel_name="Hoechst"))

        assert _names(plan.nuclei) == ["Hoechst (1)", "Hoechst (2)", "Hoechst (3)"]
        assert _names(plan.channels) == ["Hoechst (1)", "Hoechst (3)"]
