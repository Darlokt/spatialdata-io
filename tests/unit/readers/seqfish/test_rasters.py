"""seqFISH lazy raster selection, calibrated axes and label safety."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback
from ome_types.model import Channel
from spatialdata.transformations import get_transformation
from xarray import DataArray, DataTree

import spatialdata_io.readers._utils.image.tiff._dask as tiff_dask
import spatialdata_io.readers.seqfish._rasters as raster_module
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image import inspect_tiff, make_raster_options
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.seqfish._rasters import read_dapi, read_image, read_labels

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from pytest_mock import MockerFixture
    from spatialdata.transformations import BaseTransformation


def _write_dapi(path: Path, *, names: list[str] | None = None, compression: str | None = None) -> NDArray[np.uint16]:
    """Write calibrated two-channel tiled data for exact selection tests."""
    data = np.arange(2 * 48 * 64, dtype=np.uint16).reshape(2, 48, 64)
    tifffile.imwrite(
        path,
        data,
        ome=True,
        photometric="minisblack",
        tile=(16, 16),
        compression=compression,
        metadata={
            "axes": "CYX",
            "PhysicalSizeX": 0.2,
            "PhysicalSizeY": 0.5,
            "Channel": {"Name": names} if names is not None else {},
        },
    )
    return data


DEFAULT_OPTIONS = make_raster_options()
# The default 1024-pixel tiles merge the 16-pixel fixture tiles into one chunk;
# tests of bounded reads and per-task resources request the stored tiles.
NATIVE_CHUNKS = make_raster_options(chunks=(1, 16, 16))


class TestReadDapi:
    """Validate series, calibration and named axes without decoding pixels."""

    def test_malformed_tiff_retains_context(self, tmp_path: Path) -> None:
        path = tmp_path / "DAPI.tiff"
        path.write_bytes(b"not TIFF")
        with pytest.raises(ValueError, match="TIFF") as caught:
            read_dapi(ArtifactFile(path))
        assert any("seqfish" in note and str(path) in note for note in caught.value.__notes__)

    def test_multiple_series_fail(self, tmp_path: Path) -> None:
        path = tmp_path / "DAPI.tiff"
        with tifffile.TiffWriter(path) as output:
            output.write(np.zeros((16, 16), dtype=np.uint16))
            output.write(np.zeros((8, 8), dtype=np.uint16))
        with pytest.raises(ReaderFormatError, match="2 series"):
            read_dapi(ArtifactFile(path))

    @pytest.mark.parametrize(("axes", "shape"), [("ZYX", (2, 16, 16)), ("QYX", (2, 16, 16))])
    def test_non_singleton_nonspatial_axes_fail(self, tmp_path: Path, axes: str, shape: tuple[int, int, int]) -> None:
        path = tmp_path / "DAPI.tiff"
        tifffile.imwrite(path, np.zeros(shape, dtype=np.uint16), photometric="minisblack", metadata={"axes": axes})
        with pytest.raises(ReaderFormatError, match="singleton additional axes"):
            read_dapi(ArtifactFile(path))

    def test_singleton_axes_preserve_singleton_spatial_dimensions(self, tmp_path: Path) -> None:
        path = tmp_path / "DAPI.tiff"
        data = np.arange(16, dtype=np.uint16).reshape(1, 1, 1, 16)
        tifffile.imwrite(
            path,
            data,
            ome=True,
            photometric="minisblack",
            metadata={"axes": "ZCYX", "PhysicalSizeX": 0.2, "PhysicalSizeY": 0.5},
        )
        dapi = read_dapi(ArtifactFile(path))
        image = read_image(dapi, roi="Roi1", options=DEFAULT_OPTIONS)
        assert isinstance(image, DataArray)
        assert image.shape == (1, 1, 16)
        np.testing.assert_array_equal(image.data.compute(), data.reshape(1, 1, 16))

    @pytest.mark.parametrize("invalid", ["count", "ids"])
    def test_invalid_external_channel_metadata(self, tmp_path: Path, mocker: MockerFixture, invalid: str) -> None:
        path = tmp_path / "DAPI.tiff"
        _write_dapi(path)
        metadata = inspect_tiff(ArtifactFile(path))
        ome = metadata.ome
        assert ome is not None
        ome.images[0].pixels.channels = (
            [] if invalid == "count" else [Channel(id="Channel:0:0"), Channel(id="Channel:0:0")]
        )
        mocker.patch.object(raster_module, "_inspect", return_value=metadata)
        with pytest.raises(ReaderFormatError, match="OME"):
            read_dapi(ArtifactFile(path))


class TestReadImage:
    """Keep native tile I/O lazy, bounded, serializable and correctly calibrated."""

    @pytest.mark.parametrize("compression", [None, "deflate"])
    def test_native_tiles_partial_reads_and_serialization(
        self, tmp_path: Path, mocker: MockerFixture, compression: str | None
    ) -> None:
        path = tmp_path / "DAPI.tiff"
        source = _write_dapi(path, names=["DAPI", "CD3"], compression=compression)
        pixels = mocker.spy(tiff_dask, "_read_tiff_selection")
        tasks = mocker.Mock()
        with Callback(posttask=tasks):
            dapi = read_dapi(ArtifactFile(path))
            image = read_image(dapi, roi="Roi1", options=NATIVE_CHUNKS)
        assert isinstance(image, DataArray)
        tasks.assert_not_called()
        pixels.assert_not_called()
        assert image.dims == ("c", "y", "x")
        assert image.data.chunks == ((1, 1), (16, 16, 16), (16, 16, 16, 16))
        assert image.coords["c"].values.tolist() == ["DAPI", "CD3"]
        # Upstream overload does not narrow the single-coordinate-system result.
        transformation = cast("BaseTransformation", get_transformation(image, to_coordinate_system="Roi1"))
        np.testing.assert_array_equal(
            transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), np.diag([0.2, 0.5, 1])
        )
        graph = pickle.loads(pickle.dumps(image.data))  # noqa: S301, RUF100 - Test-owned graph only.
        np.testing.assert_array_equal(graph[:, :4, :6].compute(), source[:, :4, :6])
        assert pixels.call_count == 2

    @pytest.mark.parametrize("names", [None, ["same", "same"]])
    def test_missing_or_duplicate_names_use_ome_ids(self, tmp_path: Path, names: list[str] | None) -> None:
        path = tmp_path / "DAPI.tiff"
        _write_dapi(path, names=names)
        dapi = read_dapi(ArtifactFile(path))
        image = read_image(dapi, roi="Roi1", options=make_raster_options(chunks=(2, 12, 16)))
        assert isinstance(image, DataArray)
        assert image.coords["c"].values.tolist() == ["Channel:0:0", "Channel:0:1"]
        assert image.data.chunks[0] == (2,)

    def test_pyramid_generation_is_explicit_and_extent_bounded(self, tmp_path: Path) -> None:
        path = tmp_path / "DAPI.tiff"
        _write_dapi(path)
        dapi = read_dapi(ArtifactFile(path))
        pyramid = read_image(dapi, roi="Roi1", options=make_raster_options([2, 2], 16))
        assert isinstance(pyramid, DataTree)
        assert list(pyramid.children) == ["scale0", "scale1", "scale2"]
        with pytest.raises(ValueError, match="extent"):
            read_image(dapi, roi="Roi1", options=make_raster_options([64]))


class TestReadLabels:
    """Preserve integer labels and DAPI frame correspondence without eager validation."""

    @pytest.mark.parametrize("calibrated", [False, True])
    def test_ome_labels_inherit_or_confirm_dapi_calibration(self, tmp_path: Path, *, calibrated: bool) -> None:
        dapi, path = tmp_path / "DAPI.tiff", tmp_path / "Segmentation.tiff"
        _write_dapi(dapi)
        calibration = {"PhysicalSizeX": 0.2, "PhysicalSizeY": 0.5} if calibrated else {}
        tifffile.imwrite(
            path,
            np.zeros((48, 64), dtype=np.uint32),
            ome=True,
            photometric="minisblack",
            metadata={"axes": "YX", **calibration},
        )
        reference = read_dapi(ArtifactFile(dapi))
        result, _ = read_labels(
            ArtifactFile(path),
            reference=reference,
            roi="Roi1",
            options=DEFAULT_OPTIONS,
        )
        assert isinstance(result, DataArray)
        assert result.data.chunks == ((48,), (64,))
        assert get_transformation(result, to_coordinate_system="Roi1") == reference.scale

    @pytest.mark.parametrize("chunks", [16, (1, 12, 16)])
    def test_labels_keep_values_dtype_and_requested_spatial_chunks(
        self, tmp_path: Path, mocker: MockerFixture, chunks: int | tuple[int, int, int]
    ) -> None:
        dapi, path = tmp_path / "DAPI.tiff", tmp_path / "Segmentation.tiff"
        _write_dapi(dapi)
        source = np.zeros((48, 64), dtype=np.uint64)
        source[:4, :6] = 2**54 + 1
        tifffile.imwrite(path, source, tile=(16, 16), photometric="minisblack", compression="deflate")
        reference = read_dapi(ArtifactFile(dapi))
        pixels = mocker.spy(tiff_dask, "_read_tiff_selection")
        result, dtype = read_labels(
            ArtifactFile(path),
            reference=reference,
            roi="Roi1",
            options=make_raster_options(chunks=chunks),
        )
        assert isinstance(result, DataArray)
        assert dtype == np.dtype(np.uint64)
        assert result.dims == ("y", "x")
        assert max(result.data.chunks[0]) == (12 if isinstance(chunks, tuple) else 16)
        pixels.assert_not_called()
        np.testing.assert_array_equal(result.data[:4, :6].compute(), source[:4, :6])
        assert pixels.call_count == 1

    @pytest.mark.parametrize("invalid", ["shape", "dtype", "uint64", "calibration"])
    def test_invalid_labels(self, tmp_path: Path, invalid: str) -> None:
        dapi, path = tmp_path / "DAPI.tiff", tmp_path / "Segmentation.tiff"
        _write_dapi(dapi)
        shape = (47, 64) if invalid == "shape" else (48, 64)
        dtype = np.float32 if invalid == "dtype" else np.uint64 if invalid == "uint64" else np.uint32
        tifffile.imwrite(
            path,
            np.zeros(shape, dtype=dtype),
            photometric="minisblack",
            ome=invalid == "calibration",
            metadata={"axes": "YX", "PhysicalSizeX": 0.3, "PhysicalSizeY": 0.5},
        )
        reference = read_dapi(ArtifactFile(dapi))
        with pytest.raises(ValueError, match=r"extent|integer label|32-bit|calibrated pixel"):
            read_labels(
                ArtifactFile(path),
                reference=reference,
                roi="Roi1",
                options=make_raster_options((2,) if invalid == "uint64" else ()),
            )

    def test_label_pyramid_preserves_large_uint32_ids(self, tmp_path: Path) -> None:
        dapi, path = tmp_path / "DAPI.tiff", tmp_path / "Segmentation.tiff"
        _write_dapi(dapi)
        source = np.full((48, 64), np.iinfo(np.uint32).max, dtype=np.uint32)
        tifffile.imwrite(path, source, tile=(16, 16), photometric="minisblack")
        reference = read_dapi(ArtifactFile(dapi))
        result, _ = read_labels(
            ArtifactFile(path),
            reference=reference,
            roi="Roi1",
            options=make_raster_options([2, 2]),
        )
        assert isinstance(result, DataTree)
        for level in result.children.values():
            array = level["image"].data.compute()
            assert array.dtype == np.dtype(np.uint32)
            np.testing.assert_array_equal(np.unique(array), [np.iinfo(np.uint32).max])

    @pytest.mark.parametrize("fail", [False, True])
    def test_task_resources_close_on_success_and_failure(
        self, tmp_path: Path, mocker: MockerFixture, *, fail: bool
    ) -> None:
        path = tmp_path / "DAPI.tiff"
        _write_dapi(path)
        reference = read_dapi(ArtifactFile(path))
        image = read_image(reference, roi="Roi1", options=NATIVE_CHUNKS)
        assert isinstance(image, DataArray)
        with tifffile.TiffFile(path) as tiff:
            store = tiff.aszarr(series=0, level=0, squeeze=True)
            close_store = mocker.spy(store, "close")
            mocker.patch.object(tiff, "aszarr", return_value=store)
            mocker.patch.object(tiff_dask.tifffile, "TiffFile", return_value=tiff)
            if fail:
                mocker.patch.object(tiff_dask.zarr, "open_array", side_effect=RuntimeError("injected"))
                with pytest.raises(RuntimeError, match="injected"):
                    image.data[:, :4, :6].compute(scheduler="synchronous")
            else:
                image.data[:1, :4, :6].compute(scheduler="synchronous")
            close_store.assert_called_once()
            assert tiff.filehandle.closed
