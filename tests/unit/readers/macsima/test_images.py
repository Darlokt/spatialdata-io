"""Lazy padded channel stacks, transformations, pyramids, and chunks."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import pytest
from dask.callbacks import Callback
from spatialdata.models import get_channel_names
from spatialdata.transformations import Identity, Scale, get_transformation
from xarray import DataArray, DataTree

import spatialdata_io.readers._utils.image.tiff._dask as tiff_dask
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile
from spatialdata_io.readers.macsima._channels import ChannelOptions, StackPlan, plan_stack
from spatialdata_io.readers.macsima._images import make_image
from spatialdata_io.readers.macsima._layout import StackFiles
from spatialdata_io.readers.macsima._metadata import read_channel
from tests.integration.readers.macsima._fixtures import ChannelSpec, full_resolution, write_channel

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from pytest_mock import MockerFixture

_DEFAULT = make_raster_options()


def _plan(
    directory: Path, specs: tuple[ChannelSpec, ...], *, padded: bool = False
) -> tuple[StackPlan, list[NDArray[np.uint16]]]:
    pixels = [write_channel(directory, spec, version=1, index=index) for index, spec in enumerate(specs)]
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


class TestMakeImage:
    """Build one lazy CYX image without executing pixel tasks."""

    def test_stacks_exact_pixels_lazily_in_micrometers(self, tmp_path: Path) -> None:
        plan, pixels = _plan(tmp_path, (ChannelSpec("CD3", 1, "APC"), ChannelSpec("CD4", 2, "PE")))
        executed: list[object] = []

        with Callback(posttask=lambda *args: executed.append(args)):
            element = make_image(
                plan.channels, frame="run", pixel_size_um=0.17, include_transformations=True, raster=_DEFAULT
            )

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

        element = make_image(
            plan.channels, frame="run", pixel_size_um=0.17, include_transformations=True, raster=_DEFAULT
        )

        expected = np.zeros((2, 8, 9), dtype=np.uint16)
        expected[0, 2:8, 0:8] = pixels[0]
        expected[1, 0:7, 3:9] = pixels[1]
        np.testing.assert_array_equal(full_resolution(element).data.compute(), expected)

    def test_without_transformations_the_frame_is_the_pixel_grid(self, tmp_path: Path) -> None:
        plan, _ = _plan(tmp_path, (ChannelSpec("CD3", 1, "APC"),))

        element = make_image(
            plan.channels,
            frame="run",
            pixel_size_um=0.17,
            include_transformations=False,
            raster=make_raster_options(chunks=1),
        )

        frames = get_transformation(element, get_all=True)
        assert isinstance(frames, dict)
        assert list(frames) == ["run"]
        assert isinstance(frames["run"], Identity)

    def test_chunks_and_pyramid(self, tmp_path: Path) -> None:
        plan, _ = _plan(
            tmp_path, (ChannelSpec("CD3", 1, "APC", shape=(16, 20)), ChannelSpec("CD4", 2, "PE", shape=(16, 20)))
        )

        element = make_image(
            plan.channels,
            frame="run",
            pixel_size_um=0.17,
            include_transformations=True,
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
                include_transformations=True,
                raster=make_raster_options((2, 4), 1),
            )

    def test_small_region_reads_one_band_of_one_channel(self, tmp_path: Path, mocker: MockerFixture) -> None:
        plan, pixels = _plan(tmp_path, (ChannelSpec("CD3", 1, "APC"), ChannelSpec("CD4", 2, "PE")))
        element = make_image(
            plan.channels, frame="run", pixel_size_um=0.17, include_transformations=True, raster=_DEFAULT
        )
        selection = mocker.spy(tiff_dask, "_read_tiff_selection")

        np.testing.assert_array_equal(full_resolution(element).data[1, :2, :3].compute(), pixels[1][:2, :3])

        selection.assert_called_once()
