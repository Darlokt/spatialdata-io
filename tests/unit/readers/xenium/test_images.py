"""Lazy morphology images with OME channel names, calibration and linked-set checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from dask.callbacks import Callback
from spatialdata.transformations import Scale, get_transformation
from xarray import DataArray, DataTree

from spatialdata_io.readers._utils.errors import RasterFormatError, ReaderFormatError
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.xenium._images import read_morphology
from spatialdata_io.readers.xenium._layout import FocusImages
from tests.integration.readers.xenium import _fixtures as fx

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_OPTIONS = make_raster_options()


def _focus_set(directory: Path, channels: tuple[str, ...]) -> tuple[FocusImages, np.ndarray]:
    names = tuple(f"ch{index:04d}_x.ome.tif" for index in range(len(channels)))
    pixels = np.stack([np.full(fx.SHAPE, index + 1, dtype=np.uint16) for index in range(len(channels))])
    pixels[:, 3, 5] = 900
    fx.write_focus_set(directory, names, channels, pixels)
    files = tuple(ArtifactFile(directory / name) for name in names)
    return FocusImages(files[0], files), pixels


class TestReadMorphology:
    """Morphology images are lazy, calibrated and named by their OME channels."""

    @pytest.mark.parametrize("channels", [("DAPI",), fx.FOCUS_FOUR, fx.FOCUS_PROTEIN], ids=["one", "four", "protein"])
    def test_channel_sets(self, tmp_path: Path, channels: tuple[str, ...]) -> None:
        images, pixels = _focus_set(tmp_path, channels)
        tasks: list[object] = []

        with Callback(posttask=lambda *args: tasks.append(args)):
            image = read_morphology(images, pixel_size=fx.PIXEL_SIZE, raster=DEFAULT_OPTIONS)

        assert tasks == []
        assert isinstance(image, DataArray)
        assert image.dims == ("c", "y", "x")
        assert image.coords["c"].to_numpy().tolist() == list(channels)
        assert image.dtype == np.uint16
        transformation = get_transformation(image, to_coordinate_system="global")
        assert isinstance(transformation, Scale)
        np.testing.assert_array_equal(transformation.scale, [fx.PIXEL_SIZE, fx.PIXEL_SIZE])
        np.testing.assert_array_equal(image.data[:, 2:5, 4:7].compute(), pixels[:, 2:5, 4:7])

    def test_single_file_of_xoa_1(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_1_9)
        entry = ArtifactFile(tmp_path / "morphology_focus.ome.tif")

        image = read_morphology(FocusImages(entry, (entry,)), pixel_size=fx.PIXEL_SIZE, raster=DEFAULT_OPTIONS)

        assert image.coords["c"].to_numpy().tolist() == ["DAPI"]
        np.testing.assert_array_equal(image.data.compute(), bundle.focus)

    def test_default_chunks_hold_one_channel_at_every_level(self, tmp_path: Path) -> None:
        images, pixels = _focus_set(tmp_path, fx.FOCUS_FOUR)
        single = read_morphology(images, pixel_size=fx.PIXEL_SIZE, raster=DEFAULT_OPTIONS)

        pyramid = read_morphology(images, pixel_size=fx.PIXEL_SIZE, raster=make_raster_options((2,)))

        assert isinstance(single, DataArray)
        assert isinstance(pyramid, DataTree)
        # Default tiles exceed the image, so each chunk is one whole channel.
        assert single.data.chunksize == (1, *fx.SHAPE)
        assert pyramid["scale0"]["image"].data.chunks == single.data.chunks
        assert pyramid["scale1"]["image"].data.chunksize == (1, fx.SHAPE[0] // 2, fx.SHAPE[1] // 2)
        np.testing.assert_array_equal(pyramid["scale0"]["image"].data.compute(), pixels)

    def test_raster_options(self, tmp_path: Path) -> None:
        images, pixels = _focus_set(tmp_path, fx.FOCUS_FOUR)

        image = read_morphology(images, pixel_size=fx.PIXEL_SIZE, raster=make_raster_options((2,), (1, 8, 8)))

        assert isinstance(image, DataTree)
        assert image["scale0"]["image"].data.chunksize == (1, 8, 8)
        np.testing.assert_array_equal(image["scale0"]["image"].data.compute(), pixels)

    def test_pixel_size_must_match_experiment(self, tmp_path: Path) -> None:
        images, _ = _focus_set(tmp_path, ("DAPI",))

        with pytest.raises(ReaderFormatError, match=r"0\.2 µm from experiment\.xenium"):
            read_morphology(images, pixel_size=0.2, raster=DEFAULT_OPTIONS)

    def test_linked_set_must_equal_the_directory(self, tmp_path: Path) -> None:
        images, _ = _focus_set(tmp_path, fx.FOCUS_FOUR)
        extra = ArtifactFile(tmp_path / "ch9999_extra.ome.tif")
        extra.write_bytes(images.entry.read_bytes())

        with pytest.raises(ReaderFormatError, match="exactly the OME-TIFF files"):
            read_morphology(
                FocusImages(images.entry, (*images.files, extra)), pixel_size=fx.PIXEL_SIZE, raster=DEFAULT_OPTIONS
            )

    def test_missing_linked_file_keeps_context(self, tmp_path: Path) -> None:
        images, _ = _focus_set(tmp_path, fx.FOCUS_FOUR)
        images.files[2].unlink()

        with pytest.raises(RasterFormatError, match="non-file TIFF source") as caught:
            read_morphology(images, pixel_size=fx.PIXEL_SIZE, raster=DEFAULT_OPTIONS)

        assert any("xenium" in note for note in caught.value.__notes__)

    def test_channel_names_must_be_unique(self, tmp_path: Path) -> None:
        images, _ = _focus_set(tmp_path, ("DAPI", "DAPI"))

        with pytest.raises(ReaderFormatError, match="unique nonempty OME channel names"):
            read_morphology(images, pixel_size=fx.PIXEL_SIZE, raster=DEFAULT_OPTIONS)
