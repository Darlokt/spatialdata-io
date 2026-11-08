"""Lazy morphology images with OME channel names, calibration and linked-set checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback
from spatialdata.transformations import Scale, get_transformation
from xarray import DataArray, DataTree

from spatialdata_io.readers._utils.errors import RasterFormatError, ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.xenium import _images
from spatialdata_io.readers.xenium._images import read_morphology
from spatialdata_io.readers.xenium._layout import MorphologyImage
from tests.integration.readers.xenium import _fixtures as fx

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

DEFAULT_OPTIONS = make_raster_options()
CONTEXT = ReaderErrorContext("xenium", "morphology focus image", "6.0.0")


def _focus_set(directory: Path, channels: tuple[str, ...]) -> tuple[MorphologyImage, np.ndarray]:
    names = tuple(f"ch{index:04d}_x.ome.tif" for index in range(len(channels)))
    pixels = np.stack([np.full(fx.SHAPE, index + 1, dtype=np.uint16) for index in range(len(channels))])
    pixels[:, 3, 5] = 900
    fx.write_focus_set(directory, names, channels, pixels)
    return MorphologyImage(ArtifactFile(directory / names[0]), names[0], frozenset(names)), pixels


def _read(image: MorphologyImage, **options: object) -> DataArray | DataTree:
    arguments = {"pixel_size": fx.PIXEL_SIZE, "raster": DEFAULT_OPTIONS, "context": CONTEXT} | options
    return read_morphology(image, **arguments)  # pyrefly: ignore[bad-argument-type] - Test-chosen typed options.


class TestReadMorphology:
    """Morphology images are lazy, calibrated and named by their OME channels."""

    @pytest.mark.parametrize("channels", [("DAPI",), fx.FOCUS_FOUR, fx.FOCUS_PROTEIN], ids=["one", "four", "protein"])
    def test_channel_sets(self, tmp_path: Path, channels: tuple[str, ...]) -> None:
        images, pixels = _focus_set(tmp_path, channels)
        tasks: list[object] = []

        with Callback(posttask=lambda *args: tasks.append(args)):
            image = _read(images)

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

        image = _read(MorphologyImage(entry, entry.name, frozenset({entry.name})))

        assert image.coords["c"].to_numpy().tolist() == ["DAPI"]
        np.testing.assert_array_equal(image.data.compute(), bundle.focus)

    def test_default_chunks_hold_one_channel_at_every_level(self, tmp_path: Path) -> None:
        images, pixels = _focus_set(tmp_path, fx.FOCUS_FOUR)
        single = _read(images)

        pyramid = _read(images, raster=make_raster_options((2,)))

        assert isinstance(single, DataArray)
        assert isinstance(pyramid, DataTree)
        # Default tiles exceed the image, so each chunk is one whole channel.
        assert single.data.chunksize == (1, *fx.SHAPE)
        assert pyramid["scale0"]["image"].data.chunks == single.data.chunks
        assert pyramid["scale1"]["image"].data.chunksize == (1, fx.SHAPE[0] // 2, fx.SHAPE[1] // 2)
        np.testing.assert_array_equal(pyramid["scale0"]["image"].data.compute(), pixels)

    def test_raster_options(self, tmp_path: Path) -> None:
        images, pixels = _focus_set(tmp_path, fx.FOCUS_FOUR)

        image = _read(images, raster=make_raster_options((2,), (1, 8, 8)))

        assert isinstance(image, DataTree)
        assert image["scale0"]["image"].data.chunksize == (1, 8, 8)
        np.testing.assert_array_equal(image["scale0"]["image"].data.compute(), pixels)

    def test_pixel_size_must_match_experiment(self, tmp_path: Path) -> None:
        images, _ = _focus_set(tmp_path, ("DAPI",))

        with pytest.raises(ReaderFormatError, match=r"0\.2 µm from experiment\.xenium") as caught:
            _read(images, pixel_size=0.2)

        assert caught.value.context == CONTEXT

    def test_linked_set_must_equal_the_directory(self, tmp_path: Path) -> None:
        images, _ = _focus_set(tmp_path, fx.FOCUS_FOUR)
        extra = MorphologyImage(images.entry, images.name, images.members | {"ch9999_extra.ome.tif"})

        with pytest.raises(ReaderFormatError, match="exactly the OME-TIFF files"):
            _read(extra)

    def test_linked_names_are_compared_without_resolving_symlinks(self, tmp_path: Path) -> None:
        # A relaxed read of an annexed XOA 1.x file keeps the bundle name of its link.
        bundle = fx.write_xenium_bundle(tmp_path / "bundle", fx.XOA_1_2)
        link = tmp_path / "bundle" / "morphology_focus.ome.tif"
        target = tmp_path / "annex" / "SHA256E-s1--0f"
        target.parent.mkdir()
        link.rename(target)
        link.symlink_to(target)

        image = _read(MorphologyImage(ArtifactFile(link), link.name, frozenset({link.name})))

        np.testing.assert_array_equal(image.data.compute(), bundle.focus)

    def test_missing_linked_file_keeps_context(self, tmp_path: Path) -> None:
        images, _ = _focus_set(tmp_path, fx.FOCUS_FOUR)
        (tmp_path / "ch0002_x.ome.tif").unlink()

        with pytest.raises(RasterFormatError, match="non-file TIFF source") as caught:
            _read(images)

        assert any("xenium" in note for note in caught.value.__notes__)

    def test_requires_one_ome_series(self, tmp_path: Path) -> None:
        path = tmp_path / "plain.tif"
        tifffile.imwrite(path, np.zeros(fx.SHAPE, dtype=np.uint16))

        with pytest.raises(ReaderFormatError, match="one OME image series; found 1 series"):
            _read(MorphologyImage(ArtifactFile(path), path.name, frozenset({path.name})))

    def test_channel_names_must_be_unique(self, tmp_path: Path) -> None:
        images, _ = _focus_set(tmp_path, ("DAPI", "DAPI"))

        with pytest.raises(ReaderFormatError, match="unique nonempty OME channel names"):
            _read(images)

    def test_channels_must_match_their_names(self, tmp_path: Path, mocker: MockerFixture) -> None:
        images, _ = _focus_set(tmp_path, fx.FOCUS_FOUR)
        mocker.patch.object(_images, "normalize_raster_axes", return_value=da.zeros((3, *fx.SHAPE), dtype=np.uint16))

        with pytest.raises(ReaderFormatError, match="3 channels for 4 names"):
            _read(images)

    def test_scale_factors_must_fit_the_image(self, tmp_path: Path) -> None:
        images, _ = _focus_set(tmp_path, ("DAPI",))

        with pytest.raises(ValueError, match=r"exceed the spatial extent of the morphology image \(24 x 32"):
            _read(images, raster=make_raster_options((4, 8)))
