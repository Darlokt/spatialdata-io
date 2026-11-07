"""Lazy MCMICRO images and labels with explicit axes, grids, channels, and frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback
from spatialdata.transformations import BaseTransformation, Identity, Scale, get_transformation
from xarray import DataArray, DataTree

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.mcmicro._rasters import ImageGrid, read_image, read_labels, read_mask

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from spatialdata_io.readers._utils.image import RasterOptions

_DEFAULT = make_raster_options()


def _write(
    path: Path, array: np.ndarray, axes: str, *, rowsperstrip: int | None = None, compression: str | None = None
) -> ArtifactFile:
    tifffile.imwrite(
        path,
        array,
        metadata={"axes": axes},
        photometric="minisblack",
        rowsperstrip=rowsperstrip,
        compression=compression,
    )
    return ArtifactFile(path)


def _frames(element: DataArray | DataTree) -> dict[str, BaseTransformation]:
    frames = get_transformation(element, get_all=True)
    assert isinstance(frames, dict)
    return frames


def _write_ome(path: Path, array: np.ndarray, axes: str, **sizes: float) -> ArtifactFile:
    tifffile.imwrite(path, array, ome=True, photometric="minisblack", metadata={"axes": axes, **sizes})
    return ArtifactFile(path)


def _image(
    path: ArtifactFile, channels: tuple[str, ...], options: RasterOptions = _DEFAULT
) -> tuple[DataArray | DataTree, ImageGrid]:
    return read_image(path, name="core_2", calibrated_frame="core_2", channels=channels, options=options)


def _grid(shape: tuple[int, int] = (4, 5), sizes: tuple[float | None, float | None] = (None, None)) -> ImageGrid:
    return ImageGrid("image", shape, sizes, "global")


class TestReadImage:
    """Images keep source dtype, panel channels, requested chunks, and their frame."""

    def test_cyx_panel_channels_lazy_and_default_chunks(self, tmp_path: Path) -> None:
        values = np.arange(2 * 20 * 12, dtype=np.uint16).reshape(2, 20, 12)
        path = _write(tmp_path / "image.ome.tif", values, "CYX", rowsperstrip=5, compression="deflate")
        tasks: list[object] = []
        with Callback(posttask=lambda *args: tasks.append(args)):
            element, grid = _image(path, ("CD3", "DAPI"))
        assert not tasks
        assert isinstance(element, DataArray)
        assert grid.shape == (20, 12)
        assert element.dims == ("c", "y", "x")
        assert element.coords["c"].to_numpy().tolist() == ["CD3", "DAPI"]
        assert element.dtype == np.dtype(np.uint16)
        # Default chunks hold one channel each in one tile of at most 1024 pixels.
        assert element.data.chunks == ((1, 1), (20,), (12,))
        # Without OME sizes the core stays in its pixel frame, never in the micrometre frame.
        assert grid == ImageGrid("core_2", (20, 12), (None, None), "core_2")
        assert set(_frames(element)) == {"core_2_pixels"}
        assert isinstance(get_transformation(element, "core_2_pixels"), Identity)
        np.testing.assert_array_equal(element.data.compute(), values)

    def test_single_plane_gets_channel_axis(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "image.tif", np.ones((4, 5), dtype=np.uint8), "YX")
        element, grid = _image(path, ("DAPI",))
        assert grid.shape == (4, 5)
        assert element.shape == (1, 4, 5)

    def test_pyramid_and_chunks(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "image.tif", np.ones((1, 16, 16), dtype=np.uint8), "CYX")
        element, _ = _image(path, ("A",), make_raster_options((2,), (1, 4, 8)))
        assert isinstance(element, DataTree)
        assert element["scale0"]["image"].data.chunks == ((1,), (4, 4, 4, 4), (8, 8))

    def test_channel_count_must_match_panel(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "image.tif", np.ones((2, 4, 5), dtype=np.uint8), "CYX")
        with pytest.raises(ReaderFormatError, match="2 channels for 3 markers"):
            _image(path, ("A", "B", "C"))

    def test_pyramid_cannot_exceed_extent(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "image.tif", np.ones((1, 4, 5), dtype=np.uint8), "CYX")
        with pytest.raises(ValueError, match="exceed the spatial extent"):
            _image(path, ("A",), make_raster_options((2, 4)))

    def test_unnamed_extra_axis_fails(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "image.tif", np.ones((2, 2, 4, 5), dtype=np.uint8), "ZCYX")
        with pytest.raises(ReaderFormatError, match="singleton"):
            _image(path, ("A", "B"))

    def test_multiple_series_fail(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.ones((4, 5), dtype=np.uint8), metadata={"axes": "YX"})
            writer.write(np.ones((3, 3), dtype=np.uint8), metadata={"axes": "YX"})
        with pytest.raises(ReaderFormatError, match="2 TIFF series"):
            _image(ArtifactFile(path), ("A",))

    def test_corrupt_tiff_keeps_type_and_gains_context(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        path.write_bytes(b"not a tiff")
        with pytest.raises(ValueError, match="TIFF") as error:
            _image(ArtifactFile(path), ("A",))
        assert any("mcmicro" in note for note in getattr(error.value, "__notes__", []))


class TestReadLabels:
    """Segmentations are integer YX arrays in their image's frame; pixels are never read."""

    def test_singleton_axis_grid_dtype_and_frame(self, tmp_path: Path, mocker: MockerFixture) -> None:
        values = np.arange(20, dtype=np.uint32).reshape(1, 4, 5)
        path = _write(tmp_path / "cell.ome.tif", values, "CYX")
        decoder = mocker.spy(tifffile.TiffPage, "asarray")
        element, dtype = read_labels(path, grid=_grid(), options=make_raster_options(chunks=(1, 2, 3)))
        decoder.assert_not_called()
        assert dtype == np.dtype(np.uint32)
        assert element.dims == ("y", "x")
        assert element.data.chunks == ((2, 2), (3, 2))
        assert set(_frames(element)) == {"image_pixels"}
        np.testing.assert_array_equal(element.data.compute(), values[0])

    def test_inherits_image_calibration(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "cell.ome.tif", np.ones((4, 5), dtype=np.uint16), "YX")
        element, _ = read_labels(path, grid=_grid(sizes=(2.0, 3.0)), options=_DEFAULT)
        transform = get_transformation(element, to_coordinate_system="global")
        assert set(_frames(element)) == {"global"}
        assert isinstance(transform, Scale)
        np.testing.assert_array_equal(transform.scale, [2, 3])

    @pytest.mark.parametrize(
        ("dtype", "shape", "options", "match"),
        [
            (np.uint16, (4, 6), _DEFAULT, "image grid"),
            (np.float32, (4, 5), _DEFAULT, "integer labels"),
            (np.bool_, (4, 5), _DEFAULT, "integer labels"),
            (np.uint64, (4, 5), make_raster_options((2,)), "<=32-bit"),
            (np.uint16, (4, 5), make_raster_options((8,)), "exceed"),
        ],
    )
    def test_rejects(
        self,
        tmp_path: Path,
        dtype: type[np.generic],
        shape: tuple[int, int],
        options: RasterOptions,
        match: str,
    ) -> None:
        path = _write(tmp_path / "labels.tif", np.ones((4, 5), dtype), "YX")
        with pytest.raises(ValueError, match=match):
            read_labels(path, grid=_grid(shape), options=options)

    @pytest.mark.parametrize(
        ("image", "labels"),
        [((0.3, None), {"PhysicalSizeX": 0.4}), ((0.3, 0.7), {"PhysicalSizeX": 0.3, "PhysicalSizeY": 0.8})],
    )
    def test_conflicting_declared_sizes_fail(
        self, tmp_path: Path, image: tuple[float, float | None], labels: dict[str, float]
    ) -> None:
        path = _write_ome(tmp_path / "labels.ome.tif", np.ones((4, 5), dtype=np.uint8), "YX", **labels)
        with pytest.raises(ReaderFormatError, match="calibration") as caught:
            read_labels(path, grid=_grid(sizes=image), options=_DEFAULT)
        assert caught.value.context.artifact == "OME calibration"
        assert caught.value.path == path

    def test_labels_cannot_complete_a_partial_image_calibration(self, tmp_path: Path) -> None:
        path = _write_ome(
            tmp_path / "labels.ome.tif", np.ones((4, 5), dtype=np.uint8), "YX", PhysicalSizeX=0.3, PhysicalSizeY=0.7
        )
        element, _ = read_labels(path, grid=_grid(sizes=(0.3, None)), options=_DEFAULT)
        assert _frames(element) == {"image_pixels": Identity()}


class TestReadMask:
    """Tissue masks are independent labels, calibrated only by their own declaration."""

    def test_uncalibrated_mask_uses_its_pixel_frame(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "1_mask.tif", np.ones((3, 4), dtype=np.uint8), "YX")
        element = read_mask(path, name="core_1_mask", options=_DEFAULT)
        assert _frames(element) == {"core_1_mask_pixels": Identity()}

    def test_calibrated_mask_uses_own_scale(self, tmp_path: Path) -> None:
        path = _write_ome(
            tmp_path / "mask.ome.tif", np.ones((4, 5), dtype=np.uint8), "YX", PhysicalSizeX=2, PhysicalSizeY=3
        )
        element = read_mask(path, name="core_1_mask", options=_DEFAULT)
        transform = get_transformation(element, to_coordinate_system="core_1_mask")
        assert set(_frames(element)) == {"core_1_mask"}
        assert isinstance(transform, Scale)
        np.testing.assert_array_equal(transform.scale, [2, 3])

    def test_rejects_float_mask(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "1_mask.tif", np.ones((3, 4), dtype=np.float32), "YX")
        with pytest.raises(ReaderFormatError, match="integer labels"):
            read_mask(path, name="core_1_mask", options=_DEFAULT)


class TestImageGrid:
    """Only a complete declaration calibrates; one frame name never mixes units."""

    @pytest.mark.parametrize(
        ("sizes", "frame", "transform"),
        [
            ((None, None), "core_1_pixels", Identity()),
            ((0.3, None), "core_1_pixels", Identity()),
            ((None, 0.7), "core_1_pixels", Identity()),
            ((0.3, 0.7), "core_1", Scale([0.3, 0.7], axes=("x", "y"))),
        ],
    )
    def test_frame_and_transform(
        self, sizes: tuple[float | None, float | None], frame: str, transform: BaseTransformation
    ) -> None:
        grid = ImageGrid("core_1", (4, 5), sizes, "core_1")
        assert grid.frame == frame
        assert grid.transformations() == {frame: transform}
        assert grid.spacing_um == (None if None in sizes else sizes)

    def test_calibration_evidence(self) -> None:
        grid = ImageGrid("image", (4, 5), (0.3, 0.7), "global")
        assert grid.calibration == (
            "Source image OME PhysicalSizeX/PhysicalSizeY with declared source units; XY=(0.3, 0.7) micrometer/pixel"
        )

    def test_partial_sizes_are_retained(self, tmp_path: Path) -> None:
        path = _write_ome(tmp_path / "image.ome.tif", np.ones((4, 5), dtype=np.uint8), "YX", PhysicalSizeX=0.3)
        element, grid = read_image(path, name="image", calibrated_frame="global", channels=("DAPI",), options=_DEFAULT)
        assert grid == ImageGrid("image", (4, 5), (0.3, None), "global")
        assert _frames(element) == {"image_pixels": Identity()}

    def test_calibrated_image_uses_micrometre_frame(self, tmp_path: Path) -> None:
        path = _write_ome(
            tmp_path / "image.ome.tif", np.ones((4, 5), dtype=np.uint8), "YX", PhysicalSizeX=0.3, PhysicalSizeY=0.7
        )
        element, grid = read_image(path, name="image", calibrated_frame="global", channels=("DAPI",), options=_DEFAULT)
        assert grid.frame == "global"
        assert _frames(element) == {"global": Scale([0.3, 0.7], axes=("x", "y"))}
