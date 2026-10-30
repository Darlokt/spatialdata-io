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
from spatialdata_io.readers.mcmicro._rasters import ImageGrid, read_image, read_labels

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


class TestReadImage:
    """Images keep source dtype, panel channels, requested chunks, and their frame."""

    def test_cyx_panel_channels_lazy_and_default_chunks(self, tmp_path: Path) -> None:
        values = np.arange(2 * 20 * 12, dtype=np.uint16).reshape(2, 20, 12)
        path = _write(tmp_path / "image.ome.tif", values, "CYX", rowsperstrip=5, compression="deflate")
        tasks: list[object] = []
        with Callback(posttask=lambda *args: tasks.append(args)):
            element, grid = read_image(path, channels=("CD3", "DAPI"), frame="core_2", options=_DEFAULT)
        assert not tasks
        assert isinstance(element, DataArray)
        assert grid.shape == (20, 12)
        assert element.dims == ("c", "y", "x")
        assert element.coords["c"].to_numpy().tolist() == ["CD3", "DAPI"]
        assert element.dtype == np.dtype(np.uint16)
        # Default chunks hold one channel each in one tile of at most 1024 pixels.
        assert element.data.chunks == ((1, 1), (20,), (12,))
        assert set(_frames(element)) == {"core_2"}
        assert isinstance(get_transformation(element, "core_2"), Identity)
        np.testing.assert_array_equal(element.data.compute(), values)

    def test_single_plane_gets_channel_axis(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "image.tif", np.ones((4, 5), dtype=np.uint8), "YX")
        element, grid = read_image(path, channels=("DAPI",), frame="global", options=_DEFAULT)
        assert grid.shape == (4, 5)
        assert element.shape == (1, 4, 5)

    def test_pyramid_and_chunks(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "image.tif", np.ones((1, 16, 16), dtype=np.uint8), "CYX")
        element, _ = read_image(path, channels=("A",), frame="global", options=make_raster_options((2,), (1, 4, 8)))
        assert isinstance(element, DataTree)
        assert element["scale0"]["image"].data.chunks == ((1,), (4, 4, 4, 4), (8, 8))

    def test_channel_count_must_match_panel(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "image.tif", np.ones((2, 4, 5), dtype=np.uint8), "CYX")
        with pytest.raises(ReaderFormatError, match="2 channels for 3 markers"):
            read_image(path, channels=("A", "B", "C"), frame="global", options=_DEFAULT)

    def test_pyramid_cannot_exceed_extent(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "image.tif", np.ones((1, 4, 5), dtype=np.uint8), "CYX")
        with pytest.raises(ValueError, match="exceed the spatial extent"):
            read_image(path, channels=("A",), frame="global", options=make_raster_options((2, 4)))

    def test_unnamed_extra_axis_fails(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "image.tif", np.ones((2, 2, 4, 5), dtype=np.uint8), "ZCYX")
        with pytest.raises(ReaderFormatError, match="singleton"):
            read_image(path, channels=("A", "B"), frame="global", options=_DEFAULT)

    def test_multiple_series_fail(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.ones((4, 5), dtype=np.uint8), metadata={"axes": "YX"})
            writer.write(np.ones((3, 3), dtype=np.uint8), metadata={"axes": "YX"})
        with pytest.raises(ReaderFormatError, match="2 TIFF series"):
            read_image(ArtifactFile(path), channels=("A",), frame="global", options=_DEFAULT)

    def test_corrupt_tiff_keeps_type_and_gains_context(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        path.write_bytes(b"not a tiff")
        with pytest.raises(ValueError, match="TIFF") as error:
            read_image(ArtifactFile(path), channels=("A",), frame="global", options=_DEFAULT)
        assert any("mcmicro" in note for note in getattr(error.value, "__notes__", []))


class TestReadLabels:
    """Labels are integer YX arrays on their image grid; pixels are never read."""

    def test_singleton_axis_grid_dtype_and_frame(self, tmp_path: Path, mocker: MockerFixture) -> None:
        values = np.arange(20, dtype=np.uint32).reshape(1, 4, 5)
        path = _write(tmp_path / "cell.ome.tif", values, "CYX")
        decoder = mocker.spy(tifffile.TiffPage, "asarray")
        element, dtype = read_labels(
            path,
            grid=ImageGrid((4, 5), (None, None), "global", "uncalibrated"),
            frame="global",
            options=make_raster_options(chunks=(1, 2, 3)),
        )
        decoder.assert_not_called()
        assert dtype == np.dtype(np.uint32)
        assert element.dims == ("y", "x")
        assert element.data.chunks == ((2, 2), (3, 2))
        np.testing.assert_array_equal(element.data.compute(), values[0])

    def test_independent_mask_grid(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "1_mask.tif", np.ones((3, 4), dtype=np.uint8), "YX")
        element, _ = read_labels(path, grid=None, frame="core_1_mask", options=_DEFAULT)
        assert set(_frames(element)) == {"core_1_mask"}

    @pytest.mark.parametrize(
        ("dtype", "grid", "options", "match"),
        [
            (np.uint16, ImageGrid((4, 6), (None, None), "global", "uncalibrated"), _DEFAULT, "image grid"),
            (np.float32, None, _DEFAULT, "integer labels"),
            (np.uint64, None, make_raster_options((2,)), "<=32-bit"),
            (np.uint16, None, make_raster_options((8,)), "exceed"),
        ],
    )
    def test_rejects(
        self, tmp_path: Path, dtype: type[np.number], grid: ImageGrid | None, options: RasterOptions, match: str
    ) -> None:
        path = _write(tmp_path / "labels.tif", np.ones((4, 5), dtype), "YX")
        with pytest.raises(ValueError, match=match):
            read_labels(path, grid=grid, frame="global", options=options)


class TestImageGrid:
    """Reconcile partial metadata without manufacturing a complete physical frame."""

    def test_partial_sizes_retained_for_linked_grid_validation(self, tmp_path: Path) -> None:
        path = tmp_path / "image.ome.tif"
        tifffile.imwrite(path, np.ones((4, 5), dtype=np.uint8), ome=True, metadata={"axes": "YX", "PhysicalSizeX": 0.3})
        _, grid = read_image(ArtifactFile(path), channels=("DAPI",), frame="global", options=_DEFAULT)
        assert grid.declared_sizes_um == (0.3, None)
        assert grid.transform() == Identity()
        labels = tmp_path / "labels.ome.tif"
        tifffile.imwrite(
            labels, np.ones((4, 5), dtype=np.uint8), ome=True, metadata={"axes": "YX", "PhysicalSizeX": 0.4}
        )
        with pytest.raises(ReaderFormatError, match="calibration"):
            read_labels(ArtifactFile(labels), grid=grid, frame="global", options=_DEFAULT)

    def test_independent_calibrated_mask_uses_own_scale(self, tmp_path: Path) -> None:
        path = tmp_path / "mask.ome.tif"
        tifffile.imwrite(
            path,
            np.ones((4, 5), dtype=np.uint8),
            ome=True,
            metadata={"axes": "YX", "PhysicalSizeX": 2, "PhysicalSizeY": 3},
        )
        element, _ = read_labels(ArtifactFile(path), grid=None, frame="mask", options=_DEFAULT)
        transform = get_transformation(element, to_coordinate_system="mask")
        assert isinstance(transform, Scale)
        np.testing.assert_array_equal(transform.scale, [2, 3])
