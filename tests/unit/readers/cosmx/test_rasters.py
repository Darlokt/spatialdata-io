"""Lazy CosMx FOV rasters with unchanged pixels and evidenced pixel-convention frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import imagecodecs
import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback
from spatialdata.transformations import BaseTransformation, get_transformation
from xarray import DataArray, DataTree

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.cosmx._rasters import read_image, read_labels
from spatialdata_io.readers.cosmx._transforms import Placement

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.image import RasterOptions

_DEFAULT = make_raster_options()
_PLACEMENT = Placement(1, 1, 1000.0, 5000.5)
_AXES = ("x", "y")


def _frames(element: DataArray | DataTree) -> dict[str, BaseTransformation]:
    frames = get_transformation(element, get_all=True)
    assert isinstance(frames, dict)
    return frames


def _matrix(transform: BaseTransformation) -> np.ndarray:
    return transform.to_affine_matrix(input_axes=_AXES, output_axes=_AXES)


class TestReadImage:
    """Encoded and TIFF images keep source pixels; frames carry the reflection."""

    def test_png_color_composite(self, tmp_path: Path) -> None:
        pixels = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        (tmp_path / "C_F002.png").write_bytes(imagecodecs.png_encode(pixels))
        tasks: list[object] = []
        with Callback(posttask=lambda *args: tasks.append(args)):
            element, grid = read_image(
                ArtifactFile(tmp_path / "C_F002.png"), fov=2, placement=_PLACEMENT, options=_DEFAULT
            )
        assert not tasks
        assert grid == (4, 5)
        assert element.dims == ("c", "y", "x")
        assert element.coords["c"].to_numpy().tolist() == ["r", "g", "b"]
        assert element.data.chunks == ((1, 1, 1), (4,), (5,))
        np.testing.assert_array_equal(element.data.compute(), np.moveaxis(pixels, -1, 0))
        frames = _frames(element)
        assert set(frames) == {"2", "global"}
        np.testing.assert_array_equal(_matrix(frames["2"]), [[1, 0, -0.5], [0, -1, 4.5], [0, 0, 1]])
        np.testing.assert_array_equal(_matrix(frames["global"]), [[1, 0, 999.5], [0, -1, 5005.0], [0, 0, 1]])

    def test_jpeg_and_grayscale_tiff(self, tmp_path: Path) -> None:
        (tmp_path / "C_F001.jpg").write_bytes(imagecodecs.jpeg8_encode(np.zeros((4, 5, 3), np.uint8)))
        jpeg, _ = read_image(ArtifactFile(tmp_path / "C_F001.jpg"), fov=1, placement=_PLACEMENT, options=_DEFAULT)
        assert jpeg.shape == (3, 4, 5)
        tifffile.imwrite(tmp_path / "C_F001.tif", np.ones((4, 5), np.uint16))
        tiff, _ = read_image(ArtifactFile(tmp_path / "C_F001.tif"), fov=1, placement=_PLACEMENT, options=_DEFAULT)
        assert tiff.shape == (1, 4, 5)
        assert tiff.coords["c"].to_numpy().tolist() == [0]

    def test_pyramid(self, tmp_path: Path) -> None:
        tifffile.imwrite(tmp_path / "C_F001.tif", np.ones((8, 8), np.uint8))
        element, _ = read_image(
            ArtifactFile(tmp_path / "C_F001.tif"), fov=1, placement=_PLACEMENT, options=make_raster_options((2,), 4)
        )
        assert isinstance(element, DataTree)
        with pytest.raises(ValueError, match="exceed"):
            read_image(
                ArtifactFile(tmp_path / "C_F001.tif"), fov=1, placement=_PLACEMENT, options=make_raster_options((16,))
            )

    def test_multiple_series_and_corrupt_files(self, tmp_path: Path) -> None:
        with tifffile.TiffWriter(tmp_path / "C_F001.tif") as writer:
            writer.write(np.ones((4, 5), np.uint8), metadata={"axes": "YX"})
            writer.write(np.ones((3, 3), np.uint8), metadata={"axes": "YX"})
        with pytest.raises(ReaderFormatError, match="2 TIFF series"):
            read_image(ArtifactFile(tmp_path / "C_F001.tif"), fov=1, placement=_PLACEMENT, options=_DEFAULT)
        (tmp_path / "C_F002.jpg").write_bytes(b"not a jpeg")
        with pytest.raises(ValueError, match=r"JPEG|SOI") as error:
            read_image(ArtifactFile(tmp_path / "C_F002.jpg"), fov=2, placement=_PLACEMENT, options=_DEFAULT)
        assert any("cosmx" in note for note in getattr(error.value, "__notes__", []))

    def test_unnamed_extra_axis_fails(self, tmp_path: Path) -> None:
        tifffile.imwrite(tmp_path / "C_F001.tif", np.ones((2, 2, 4, 5), np.uint8), metadata={"axes": "ZCYX"})
        with pytest.raises(ReaderFormatError, match="singleton"):
            read_image(ArtifactFile(tmp_path / "C_F001.tif"), fov=1, placement=_PLACEMENT, options=_DEFAULT)


class TestReadLabels:
    """Labels are integer YX arrays on their image grid with the same frames."""

    def test_grid_dtype_frames(self, tmp_path: Path) -> None:
        values = np.arange(20, dtype=np.uint16).reshape(4, 5)
        tifffile.imwrite(tmp_path / "L_F002.tif", values)
        element, dtype = read_labels(
            ArtifactFile(tmp_path / "L_F002.tif"),
            fov=2,
            grid=(4, 5),
            placement=_PLACEMENT,
            options=make_raster_options(chunks=(1, 2, 3)),
        )
        assert dtype == np.dtype(np.uint16)
        assert element.data.chunks == ((2, 2), (3, 2))
        np.testing.assert_array_equal(element.data.compute(), values)
        np.testing.assert_array_equal(_matrix(_frames(element)["2"]), [[1, 0, -0.5], [0, -1, 4.5], [0, 0, 1]])

    def test_default_single_scale_keeps_64_bit_labels(self, tmp_path: Path) -> None:
        values = np.array([[0, 2**40], [7, 0]], dtype=np.uint64)
        tifffile.imwrite(tmp_path / "L_F001.tif", values)
        element, dtype = read_labels(
            ArtifactFile(tmp_path / "L_F001.tif"), fov=1, grid=(2, 2), placement=_PLACEMENT, options=_DEFAULT
        )
        assert isinstance(element, DataArray)
        assert dtype == np.dtype(np.uint64)
        assert element.data.chunks == ((2,), (2,))
        np.testing.assert_array_equal(element.data.compute(), values)

    @pytest.mark.parametrize(
        ("dtype", "grid", "options", "match"),
        [
            (np.uint16, (4, 6), _DEFAULT, "FOV image grid"),
            (np.float32, None, _DEFAULT, "integer labels"),
            (np.uint64, None, make_raster_options((2,)), "<=32-bit"),
        ],
    )
    def test_rejects(
        self, tmp_path: Path, dtype: type[np.number], grid: tuple[int, int] | None, options: RasterOptions, match: str
    ) -> None:
        tifffile.imwrite(tmp_path / "L_F001.tif", np.ones((4, 5), dtype))
        with pytest.raises(ValueError, match=match):
            read_labels(ArtifactFile(tmp_path / "L_F001.tif"), fov=1, grid=grid, placement=_PLACEMENT, options=options)
