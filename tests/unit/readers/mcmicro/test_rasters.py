"""Lazy MCMICRO images and labels with explicit axes, grids, channels, and frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback
from spatialdata.transformations import BaseTransformation, Identity, get_transformation
from xarray import DataArray, DataTree

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.mcmicro._rasters import RasterOptions, make_raster_options, read_image, read_labels

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

_NATIVE = RasterOptions(None, None)


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


class TestMakeRasterOptions:
    """Caller options are validated once and copied into immutable state."""

    def test_valid(self) -> None:
        factors = [2, 3]
        options = make_raster_options(factors, (1, 8, 16))
        factors.append(4)
        assert options == RasterOptions((2, 3), (1, 8, 16))
        assert make_raster_options(None, 32) == RasterOptions(None, 32)

    @pytest.mark.parametrize(
        ("factors", "chunks", "error"),
        [
            ("22", None, TypeError),
            (2, None, TypeError),
            ([], None, ValueError),
            ([1], None, ValueError),
            ([True], None, ValueError),
            (None, (1, 2), ValueError),
            (None, (1, 2, 1.5), TypeError),
            (None, True, TypeError),
            (None, 0, ValueError),
        ],
    )
    def test_rejects(self, factors: object, chunks: object, error: type[Exception]) -> None:
        with pytest.raises(error):
            make_raster_options(factors, chunks)  # pyrefly: ignore[bad-argument-type] - Invalid runtime inputs.


class TestReadImage:
    """Images keep source dtype, panel channels, storage-aligned chunks, and their frame."""

    def test_cyx_panel_channels_lazy_and_storage_aligned_chunks(self, tmp_path: Path) -> None:
        values = np.arange(2 * 20 * 12, dtype=np.uint16).reshape(2, 20, 12)
        path = _write(tmp_path / "image.ome.tif", values, "CYX", rowsperstrip=5, compression="deflate")
        tasks: list[object] = []
        with Callback(posttask=lambda *args: tasks.append(args)):
            element, grid = read_image(path, channels=("CD3", "DAPI"), frame="core_2", options=_NATIVE)
        assert not tasks
        assert isinstance(element, DataArray)
        assert grid == (20, 12)
        assert element.dims == ("c", "y", "x")
        assert element.coords["c"].to_numpy().tolist() == ["CD3", "DAPI"]
        assert element.dtype == np.dtype(np.uint16)
        # Whole 5-row strips form one row band per channel.
        assert element.data.chunks == ((1, 1), (20,), (12,))
        assert set(_frames(element)) == {"core_2"}
        assert isinstance(get_transformation(element, "core_2"), Identity)
        np.testing.assert_array_equal(element.data.compute(), values)

    def test_single_plane_gets_channel_axis(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "image.tif", np.ones((4, 5), dtype=np.uint8), "YX")
        element, grid = read_image(path, channels=("DAPI",), frame="global", options=_NATIVE)
        assert grid == (4, 5)
        assert element.shape == (1, 4, 5)

    def test_pyramid_and_chunks(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "image.tif", np.ones((1, 16, 16), dtype=np.uint8), "CYX")
        element, _ = read_image(path, channels=("A",), frame="global", options=RasterOptions((2,), (1, 4, 8)))
        assert isinstance(element, DataTree)
        assert element["scale0"]["image"].data.chunks == ((1,), (4, 4, 4, 4), (8, 8))

    def test_channel_count_must_match_panel(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "image.tif", np.ones((2, 4, 5), dtype=np.uint8), "CYX")
        with pytest.raises(ReaderFormatError, match="2 channels for 3 markers"):
            read_image(path, channels=("A", "B", "C"), frame="global", options=_NATIVE)

    def test_pyramid_cannot_exceed_extent(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "image.tif", np.ones((1, 4, 5), dtype=np.uint8), "CYX")
        with pytest.raises(ValueError, match="exceed the spatial extent"):
            read_image(path, channels=("A",), frame="global", options=RasterOptions((2, 4), None))

    def test_unnamed_extra_axis_fails(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "image.tif", np.ones((2, 2, 4, 5), dtype=np.uint8), "ZCYX")
        with pytest.raises(ReaderFormatError, match="singleton"):
            read_image(path, channels=("A", "B"), frame="global", options=_NATIVE)

    def test_multiple_series_fail(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.ones((4, 5), dtype=np.uint8), metadata={"axes": "YX"})
            writer.write(np.ones((3, 3), dtype=np.uint8), metadata={"axes": "YX"})
        with pytest.raises(ReaderFormatError, match="2 TIFF series"):
            read_image(ArtifactFile(path), channels=("A",), frame="global", options=_NATIVE)

    def test_corrupt_tiff_keeps_type_and_gains_context(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        path.write_bytes(b"not a tiff")
        with pytest.raises(ValueError, match="TIFF") as error:
            read_image(ArtifactFile(path), channels=("A",), frame="global", options=_NATIVE)
        assert any("mcmicro" in note for note in getattr(error.value, "__notes__", []))


class TestReadLabels:
    """Labels are integer YX arrays on their image grid; pixels are never read."""

    def test_singleton_axis_grid_dtype_and_frame(self, tmp_path: Path, mocker: MockerFixture) -> None:
        values = np.arange(20, dtype=np.uint32).reshape(1, 4, 5)
        path = _write(tmp_path / "cell.ome.tif", values, "CYX")
        decoder = mocker.spy(tifffile.TiffPage, "asarray")
        element, dtype = read_labels(path, grid=(4, 5), frame="global", options=RasterOptions(None, (1, 2, 3)))
        decoder.assert_not_called()
        assert dtype == np.dtype(np.uint32)
        assert element.dims == ("y", "x")
        assert element.data.chunks == ((2, 2), (3, 2))
        np.testing.assert_array_equal(element.data.compute(), values[0])

    def test_independent_mask_grid(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "1_mask.tif", np.ones((3, 4), dtype=np.uint8), "YX")
        element, _ = read_labels(path, grid=None, frame="core_1_mask", options=_NATIVE)
        assert set(_frames(element)) == {"core_1_mask"}

    @pytest.mark.parametrize(
        ("dtype", "grid", "options", "match"),
        [
            (np.uint16, (4, 6), _NATIVE, "image grid"),
            (np.float32, None, _NATIVE, "integer labels"),
            (np.uint64, None, RasterOptions((2,), None), "<=32-bit"),
            (np.uint16, None, RasterOptions((8,), None), "exceed"),
        ],
    )
    def test_rejects(
        self, tmp_path: Path, dtype: type[np.number], grid: tuple[int, int] | None, options: RasterOptions, match: str
    ) -> None:
        path = _write(tmp_path / "labels.tif", np.ones((4, 5), dtype), "YX")
        with pytest.raises(ValueError, match=match):
            read_labels(path, grid=grid, frame="global", options=options)
