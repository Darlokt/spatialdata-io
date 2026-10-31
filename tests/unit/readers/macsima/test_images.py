"""Lazy padded channel stacks, transformations, pyramids, and chunks."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal

import cloudpickle
import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback
from spatialdata.models import get_channel_names
from spatialdata.transformations import Scale, get_transformation
from xarray import DataArray, DataTree

import spatialdata_io.readers._utils.image.tiff._dask as tiff_dask
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile
from spatialdata_io.readers.macsima._channels import ChannelOptions, StackPlan, plan_stack
from spatialdata_io.readers.macsima._images import make_image
from spatialdata_io.readers.macsima._layout import StackFiles
from spatialdata_io.readers.macsima._metadata import read_channel
from tests.integration.readers.macsima._fixtures import (
    ChannelSpec,
    channel_ome_xml,
    channel_pixels,
    default_file_name,
    full_resolution,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from pytest_mock import MockerFixture

    from spatialdata_io.readers._utils.image.tiff._types import _TiffReadSpec

_DEFAULT = make_raster_options()


def _plan(
    directory: Path,
    specs: tuple[ChannelSpec, ...],
    *,
    padded: bool = False,
    storage: Literal["rows", "strip", "tiles"] = "rows",
) -> tuple[StackPlan, list[NDArray[np.uint16]]]:
    pixels = [channel_pixels(spec, index) for index, spec in enumerate(specs)]
    for index, (spec, plane) in enumerate(zip(specs, pixels, strict=True)):
        file_name = default_file_name(spec)
        tifffile.imwrite(
            directory / file_name,
            plane,
            description=channel_ome_xml(spec, version=1, file_name=file_name, index=index).encode(),
            metadata=None,
            photometric="minisblack",
            compression=None if storage == "rows" else "deflate",
            rowsperstrip=None if storage == "tiles" else spec.shape[0] if storage == "strip" else 1,
            tile=(16, 16) if storage == "tiles" else None,
        )
    sources = []
    for path in sorted(directory.iterdir()):
        source = read_channel(ArtifactFile(path.resolve()))
        assert source is not None
        sources.append(source)
    options = ChannelOptions(
        exclude_cycles=frozenset(),
        include_cycle_in_channel_names=False,
        nuclei_channel_name="DAPI",
        split_threshold_nuclei_channel=2,
    )
    stack = StackFiles(name="run", directory=ArtifactDirectory(directory.resolve()), files=())
    with pytest.warns(UserWarning, match="Zero-padding") if padded else nullcontext():
        plan = plan_stack(stack, sources, options)
    return plan, pixels


def _record_selections(mocker: MockerFixture) -> list[tuple[Path, tuple[slice, ...], int]]:
    reads: list[tuple[Path, tuple[slice, ...], int]] = []
    read_selection = tiff_dask._read_tiff_selection

    def record(spec: _TiffReadSpec, selection: tuple[slice, ...]) -> NDArray[np.generic]:
        result = read_selection(spec, selection)
        reads.append((spec.path, selection, result.nbytes))
        return result

    mocker.patch.object(tiff_dask, "_read_tiff_selection", side_effect=record)
    return reads


class TestMakeImage:
    """Build one lazy CYX image without executing pixel tasks."""

    def test_stacks_exact_pixels_lazily_in_micrometers(self, tmp_path: Path) -> None:
        plan, pixels = _plan(tmp_path, (ChannelSpec("CD3", 1, "APC"), ChannelSpec("CD4", 2, "PE")))
        executed: list[object] = []

        with Callback(posttask=lambda *args: executed.append(args)):
            element = make_image(plan.channels, frame="run", pixel_size_um=0.17, raster=_DEFAULT)

        assert not executed
        assert isinstance(element, DataArray)
        assert element.dims == ("c", "y", "x")
        assert element.dtype == np.dtype(np.uint16)
        assert get_channel_names(element) == ["CD3", "CD4"]
        transformation = get_transformation(element, to_coordinate_system="run")
        assert isinstance(transformation, Scale)
        np.testing.assert_array_equal(transformation.scale, [0.17, 0.17])
        np.testing.assert_array_equal(element.data.compute(), np.stack(pixels))

    def test_pads_unequal_shapes_and_offsets_with_zeros(self, tmp_path: Path) -> None:
        specs = (
            ChannelSpec("CD3", 1, "APC", shape=(6, 8), offset=(2, 0)),
            ChannelSpec("CD4", 2, "PE", shape=(7, 6), offset=(0, 3)),
        )
        plan, pixels = _plan(tmp_path, specs, padded=True)

        element = make_image(plan.channels, frame="run", pixel_size_um=0.17, raster=_DEFAULT)

        expected = np.zeros((2, 8, 9), dtype=np.uint16)
        expected[0, 2:8, 0:8] = pixels[0]
        expected[1, 0:7, 3:9] = pixels[1]
        np.testing.assert_array_equal(full_resolution(element).data.compute(), expected)

    def test_explicit_frame_retains_physical_scale(self, tmp_path: Path) -> None:
        plan, _ = _plan(tmp_path, (ChannelSpec("CD3", 1, "APC"),))

        element = make_image(
            plan.channels,
            frame="run",
            pixel_size_um=0.17,
            raster=make_raster_options(chunks=1),
        )

        frames = get_transformation(element, get_all=True)
        assert isinstance(frames, dict)
        assert list(frames) == ["run"]
        assert isinstance(frames["run"], Scale)

    def test_chunks_and_pyramid(self, tmp_path: Path) -> None:
        plan, _ = _plan(
            tmp_path, (ChannelSpec("CD3", 1, "APC", shape=(16, 20)), ChannelSpec("CD4", 2, "PE", shape=(16, 20)))
        )

        element = make_image(
            plan.channels,
            frame="run",
            pixel_size_um=0.17,
            raster=make_raster_options((2, 2), (2, 8, 8)),
        )

        assert isinstance(element, DataTree)
        assert list(element.children) == ["scale0", "scale1", "scale2"]
        assert full_resolution(element).data.chunks == ((2,), (8, 8), (8, 8, 4))

    def test_rejects_factors_beyond_the_extent(self, tmp_path: Path) -> None:
        plan, _ = _plan(tmp_path, (ChannelSpec("CD3", 1, "APC", shape=(6, 8)),))

        with pytest.raises(ValueError, match=r"exceed the spatial extent of MACSima stack 'run'"):
            make_image(
                plan.channels,
                frame="run",
                pixel_size_um=0.17,
                raster=make_raster_options((2, 4), 1),
            )

    def test_default_rechunking_reads_neighboring_bands_of_selected_channel(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        # Three 600-row source bands straddle the default 1024-row output boundary.
        mocker.patch.object(tiff_dask, "STRIP_BAND_BYTES", 600 * 17 * 2)
        plan, pixels = _plan(
            tmp_path,
            (ChannelSpec("CD3", 1, "APC", shape=(1800, 17)), ChannelSpec("CD4", 2, "PE", shape=(1800, 17))),
        )
        reads = _record_selections(mocker)
        executed: list[object] = []

        with Callback(posttask=lambda *args: executed.append(args)):
            element = make_image(plan.channels, frame="run", pixel_size_um=0.17, raster=_DEFAULT)
            restored = cloudpickle.loads(cloudpickle.dumps(full_resolution(element).data))

        assert not executed
        assert not reads
        assert restored.chunks == ((1, 1), (1024, 776), (17,))
        assert get_channel_names(element) == ["CD3", "CD4"]
        np.testing.assert_array_equal(
            restored[1, 1100:1106, 2:5].compute(scheduler="synchronous"), pixels[1][1100:1106, 2:5]
        )

        path = plan.channels[1].source.tiff.path
        assert sorted(reads, key=lambda read: read[1][0].start) == [
            (path, (slice(600, 1200), slice(0, 17)), 600 * 17 * 2),
            (path, (slice(1200, 1800), slice(0, 17)), 600 * 17 * 2),
        ]
        assert sum(read[2] for read in reads) == 40_800

    @pytest.mark.parametrize(
        ("storage", "expected_selection", "decoded_bytes"),
        [
            ("strip", (slice(0, 48), slice(0, 32)), 48 * 32 * 2),
            ("tiles", (slice(16, 32), slice(16, 32)), 16 * 16 * 2),
        ],
    )
    def test_compressed_storage_composes_serializable_lazy_channels(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        storage: Literal["strip", "tiles"],
        expected_selection: tuple[slice, slice],
        decoded_bytes: int,
    ) -> None:
        # A 3072-byte native strip remains indivisible above the 1024-byte target.
        mocker.patch.object(tiff_dask, "STRIP_BAND_BYTES", 1024)
        plan, pixels = _plan(
            tmp_path,
            (ChannelSpec("CD3", 1, "APC", shape=(48, 32)), ChannelSpec("CD4", 2, "PE", shape=(48, 32))),
            storage=storage,
        )
        reads = _record_selections(mocker)
        executed: list[object] = []

        with Callback(posttask=lambda *args: executed.append(args)):
            element = make_image(
                plan.channels, frame="run", pixel_size_um=0.17, raster=make_raster_options(chunks=(1, 16, 16))
            )
            restored = cloudpickle.loads(cloudpickle.dumps(full_resolution(element).data))

        assert not executed
        assert not reads
        assert restored.chunks == ((1, 1), (16, 16, 16), (16, 16))
        np.testing.assert_array_equal(
            restored[1, 18:20, 18:21].compute(scheduler="synchronous"), pixels[1][18:20, 18:21]
        )

        assert reads == [(plan.channels[1].source.tiff.path, expected_selection, decoded_bytes)]
