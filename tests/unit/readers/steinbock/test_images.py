"""Steinbock OME channels, explicit chunks, exact labels, and lazy TIFF reads."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback
from xarray import DataArray, DataTree

import spatialdata_io.readers._utils.image.tiff._dask as tiff_dask
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image import DEFAULT_CHUNKS, make_raster_options
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.steinbock._images import _read_raster, _read_rasters
from spatialdata_io.readers.steinbock._layout import _SampleLayout

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from pytest_mock import MockerFixture


def _write_sample(root: Path) -> tuple[_SampleLayout, NDArray[np.uint16], NDArray[np.uint16]]:
    """Write independent OME channel and integer mask expectations in one pixel frame."""
    image = np.arange(2 * 48 * 64, dtype=np.uint16).reshape(2, 48, 64)
    labels = np.zeros((48, 64), dtype=np.uint16)
    labels[1:4, 2:6] = 7
    image_path, label_path = root / "sample.ome.tiff", root / "sample.tiff"
    tifffile.imwrite(
        image_path,
        image,
        ome=True,
        photometric="minisblack",
        tile=(16, 16),
        metadata={"axes": "CYX", "Channel": {"Name": ["DNA", "CD3"]}},
    )
    tifffile.imwrite(label_path, labels, photometric="minisblack", tile=(16, 16))
    return _SampleLayout("sample", ArtifactFile(image_path), ArtifactFile(label_path)), image, labels


class TestReadRaster:
    """Require declared scientific axes and one full-resolution TIFF series."""

    def test_malformed_tiff_context(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.tiff"
        path.write_bytes(b"not TIFF")

        with pytest.raises(ValueError, match="TIFF") as caught:
            _read_raster(ArtifactFile(path), target_axes=("c", "y", "x"))

        assert any("steinbock" in note and str(path) in note for note in caught.value.__notes__)

    def test_multiple_series_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "multiple.tiff"
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.zeros((8, 10), dtype=np.uint16))
            writer.write(np.ones((6, 7), dtype=np.uint16))

        with pytest.raises(ReaderFormatError, match="2 series"):
            _read_raster(ArtifactFile(path), target_axes=("c", "y", "x"))

    @pytest.mark.parametrize("axes", ["QYX", "ZYX"])
    def test_nonspatial_stack_axes_rejected(self, tmp_path: Path, axes: str) -> None:
        path = tmp_path / "stack.tiff"
        tifffile.imwrite(path, np.zeros((2, 8, 10), dtype=np.uint16), photometric="minisblack", metadata={"axes": axes})

        with pytest.raises(ReaderFormatError, match="length 2"):
            _read_raster(ArtifactFile(path), target_axes=("c", "y", "x"))


class TestReadRasters:
    """Prove channels, masks, graph bounds, pyramid safety, and sample coordinates."""

    @pytest.mark.parametrize("invalid", ["shape", "float", "factor", "uint64", "int64", "no_ome"])
    def test_invalid_contract_fails_before_pixels(self, tmp_path: Path, mocker: MockerFixture, invalid: str) -> None:
        sample, image, _ = _write_sample(tmp_path)
        dtype = (
            np.float32
            if invalid == "float"
            else np.uint64
            if invalid == "uint64"
            else np.int64
            if invalid == "int64"
            else np.uint16
        )
        shape = (4, 5) if invalid == "shape" else (48, 64)
        tifffile.imwrite(sample.labels, np.zeros(shape, dtype=dtype), photometric="minisblack")
        if invalid == "no_ome":
            tifffile.imwrite(sample.image, image, ome=False, photometric="minisblack", metadata={"axes": "CYX"})

        factors = (128,) if invalid == "factor" else (2,) if invalid in {"uint64", "int64"} else ()
        pixels = mocker.spy(tiff_dask, "_read_tiff_selection")

        with pytest.raises(ValueError, match=r"extent|integer mask|32-bit|OME metadata"):
            _read_rasters(sample, make_raster_options(factors), metadata=None, calibrated_frame=sample.name)
        pixels.assert_not_called()

    @pytest.mark.parametrize("names", [["DNA", "CD3"], ["DNA", "DNA"], ["", "CD3"], None])
    def test_ome_channel_names_and_id_fallback(self, tmp_path: Path, names: list[str] | None) -> None:
        sample, image, _ = _write_sample(tmp_path)
        metadata = {"axes": "CYX"} if names is None else {"axes": "CYX", "Channel": {"Name": names}}
        tifffile.imwrite(sample.image, image, ome=True, photometric="minisblack", metadata=metadata)

        result = _read_rasters(sample, make_raster_options(), metadata=None, calibrated_frame=sample.name)

        assert isinstance(result.image, DataArray)
        expected = names if names and all(names) and len(set(names)) == len(names) else ["Channel:0:0", "Channel:0:1"]
        assert result.image.coords["c"].to_numpy().tolist() == expected
        np.testing.assert_array_equal(result.image.data.compute(), image)

    @pytest.mark.parametrize("channels", [3, 4])
    def test_scientific_channels_are_not_rgb(self, tmp_path: Path, channels: int) -> None:
        sample, _, _ = _write_sample(tmp_path)
        image = np.arange(channels * 48 * 64, dtype=np.uint16).reshape(channels, 48, 64)
        names = [f"marker-{channel}" for channel in range(channels)]
        tifffile.imwrite(
            sample.image,
            image,
            ome=True,
            photometric="minisblack",
            metadata={"axes": "CYX", "Channel": {"Name": names}},
        )

        result = _read_rasters(sample, make_raster_options(), metadata=None, calibrated_frame=sample.name)

        assert isinstance(result.image, DataArray)
        assert result.image.coords["c"].to_numpy().tolist() == names
        np.testing.assert_array_equal(result.image.data.compute(), image)

    @pytest.mark.parametrize("shape", [(1, 32, 40), (1, 1, 40), (1, 32, 1)])
    def test_singleton_spatial_and_channel_axes_preserved(self, tmp_path: Path, shape: tuple[int, int, int]) -> None:
        sample, _, _ = _write_sample(tmp_path)
        image = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        labels = np.zeros(shape[-2:], dtype=np.uint16)
        tifffile.imwrite(
            sample.image, image, ome=True, photometric="minisblack", planarconfig="separate", metadata={"axes": "CYX"}
        )
        tifffile.imwrite(sample.labels, labels, photometric="minisblack")

        result = _read_rasters(sample, make_raster_options(), metadata=None, calibrated_frame=sample.name)

        assert isinstance(result.image, DataArray)
        assert isinstance(result.labels, DataArray)
        assert result.image.shape == shape
        assert result.labels.shape == shape[-2:]
        np.testing.assert_array_equal(result.image.data.compute(), image)
        np.testing.assert_array_equal(result.labels.data.compute(), labels)

    @pytest.mark.parametrize("compression", [None, "deflate"])
    def test_native_partial_reads_are_lazy_and_standard_pickle_safe(
        self, tmp_path: Path, mocker: MockerFixture, compression: str | None
    ) -> None:
        sample, image, labels = _write_sample(tmp_path)
        tifffile.imwrite(
            sample.image,
            image,
            ome=True,
            photometric="minisblack",
            tile=(16, 16),
            compression=compression,
            metadata={"axes": "CYX"},
        )
        tifffile.imwrite(sample.labels, labels, photometric="minisblack", tile=(16, 16), compression=compression)
        pixels = mocker.spy(tiff_dask, "_read_tiff_selection")
        tasks = mocker.Mock()

        # Chunks equal to the 16-pixel tiles keep one task per stored tile.
        with Callback(posttask=tasks):
            result = _read_rasters(
                sample, make_raster_options(chunks=(1, 16, 16)), metadata=None, calibrated_frame=sample.name
            )
        tasks.assert_not_called()
        pixels.assert_not_called()
        assert isinstance(result.image, DataArray)
        assert isinstance(result.labels, DataArray)

        restored = pickle.loads(pickle.dumps(result.image.data))  # noqa: S301, RUF100 - Test-owned graph.

        assert restored.chunks == ((1, 1), (16, 16, 16), (16, 16, 16, 16))
        np.testing.assert_array_equal(restored[:1, :4, :6].compute(scheduler="synchronous"), image[:1, :4, :6])
        assert pixels.call_count == 1
        assert pixels.call_args.args[1] == (slice(0, 1), slice(0, 16), slice(0, 16))
        np.testing.assert_array_equal(result.labels.data[:4, :6].compute(scheduler="synchronous"), labels[:4, :6])
        assert pixels.call_count == 2
        assert result.grid.shape == (48, 64)
        assert result.grid.dtype == np.dtype(np.uint16)

    @pytest.mark.parametrize(
        ("chunks", "image_chunks", "label_chunks"),
        [(DEFAULT_CHUNKS, (1, 48, 64), (48, 64)), (8, (2, 8, 8), (8, 8)), ((1, 8, 16), (1, 8, 16), (8, 16))],
        ids=["default", "size", "cyx"],
    )
    def test_requested_chunks_keep_corresponding_label_axes(
        self,
        tmp_path: Path,
        chunks: int | tuple[int, int, int],
        image_chunks: tuple[int, int, int],
        label_chunks: tuple[int, int],
    ) -> None:
        sample, _, _ = _write_sample(tmp_path)

        result = _read_rasters(sample, make_raster_options(chunks=chunks), metadata=None, calibrated_frame=sample.name)

        assert isinstance(result.image, DataArray)
        assert isinstance(result.labels, DataArray)
        assert result.image.data.chunksize == image_chunks
        assert result.labels.data.chunksize == label_chunks

    @pytest.mark.parametrize("dtype", [np.uint16, np.uint32, np.int32])
    @pytest.mark.parametrize("factors", [(2, 2, 2), (3, 2)])
    def test_label_pyramids_preserve_extreme_ids(
        self, tmp_path: Path, dtype: type[np.uint16 | np.uint32 | np.int32], factors: tuple[int, ...]
    ) -> None:
        sample, _, _ = _write_sample(tmp_path)
        identifiers = np.asarray([1, 3, np.iinfo(dtype).max - 1, np.iinfo(dtype).max], dtype=dtype)
        labels = np.repeat(np.repeat(identifiers.reshape(2, 2), 32, axis=0), 32, axis=1)
        tifffile.imwrite(
            sample.image,
            np.zeros((2, 64, 64), dtype=np.uint16),
            ome=True,
            photometric="minisblack",
            metadata={"axes": "CYX"},
            tile=(16, 16),
        )
        tifffile.imwrite(sample.labels, labels, photometric="minisblack", tile=(16, 16))

        result = _read_rasters(sample, make_raster_options(factors), metadata=None, calibrated_frame=sample.name)

        assert isinstance(result.image, DataTree)
        assert isinstance(result.labels, DataTree)
        assert len(result.labels.children) == len(factors) + 1
        for level in result.labels.children.values():
            values = level["image"].data.compute()
            assert values.dtype == np.dtype(dtype)
            np.testing.assert_array_equal(np.unique(values), identifiers)
            np.testing.assert_array_equal(values[np.ix_([0, -1], [0, -1])], identifiers.reshape(2, 2))
