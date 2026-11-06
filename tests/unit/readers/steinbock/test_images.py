"""Steinbock OME channels, explicit chunks, exact labels, and lazy TIFF reads."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback
from spatialdata.transformations import get_transformation
from xarray import DataArray, DataTree

import spatialdata_io.readers._utils.image.tiff._dask as tiff_dask
import spatialdata_io.readers.steinbock._images as steinbock_images
from spatialdata_io._raster_defaults import DEFAULT_CHUNKS
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import inspect_tiff, make_raster_options
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.steinbock._images import _calibration, _read_raster, read_rasters
from spatialdata_io.readers.steinbock._layout import SampleLayout
from spatialdata_io.readers.steinbock._schemas import ImageMetadata

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import NDArray
    from pytest_mock import MockerFixture

    from spatialdata_io.readers._utils.image import TiffMetadata

_CONTEXT = ReaderErrorContext("steinbock", "OME image")


def _write_sample(root: Path) -> tuple[SampleLayout, NDArray[np.uint16], NDArray[np.uint16]]:
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
    return SampleLayout("sample", ArtifactFile(image_path), ArtifactFile(label_path)), image, labels


class TestReadRaster:
    """Require declared scientific axes and one full-resolution TIFF series."""

    def test_malformed_tiff_context(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.tiff"
        path.write_bytes(b"not TIFF")

        with pytest.raises(ValueError, match="TIFF") as caught:
            _read_raster(ArtifactFile(path), target_axes=("c", "y", "x"), context=_CONTEXT)

        assert any("steinbock" in note and str(path) in note for note in caught.value.__notes__)

    def test_multiple_series_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "multiple.tiff"
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.zeros((8, 10), dtype=np.uint16))
            writer.write(np.ones((6, 7), dtype=np.uint16))

        with pytest.raises(ReaderFormatError, match="2 series"):
            _read_raster(ArtifactFile(path), target_axes=("c", "y", "x"), context=_CONTEXT)

    @pytest.mark.parametrize("axes", ["QYX", "ZYX"])
    def test_nonspatial_stack_axes_rejected(self, tmp_path: Path, axes: str) -> None:
        path = tmp_path / "stack.tiff"
        tifffile.imwrite(path, np.zeros((2, 8, 10), dtype=np.uint16), photometric="minisblack", metadata={"axes": axes})

        with pytest.raises(ReaderFormatError, match="length 2"):
            _read_raster(ArtifactFile(path), target_axes=("c", "y", "x"), context=_CONTEXT)


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
            read_rasters(sample, make_raster_options(factors), metadata=None, calibrated_frame=sample.name)
        pixels.assert_not_called()

    @pytest.mark.parametrize("names", [["DNA", "CD3"], ["DNA", "DNA"], ["", "CD3"], None])
    def test_ome_channel_names_and_id_fallback(self, tmp_path: Path, names: list[str] | None) -> None:
        sample, image, _ = _write_sample(tmp_path)
        metadata = {"axes": "CYX"} if names is None else {"axes": "CYX", "Channel": {"Name": names}}
        tifffile.imwrite(sample.image, image, ome=True, photometric="minisblack", metadata=metadata)

        result = read_rasters(sample, make_raster_options(), metadata=None, calibrated_frame=sample.name)

        assert isinstance(result.image, DataArray)
        expected = names if names and all(names) and len(set(names)) == len(names) else ["Channel:0:0", "Channel:0:1"]
        assert result.image.coords["c"].to_numpy().tolist() == expected
        np.testing.assert_array_equal(result.image.data.compute(), image)

    @pytest.mark.parametrize("planar", [False, True], ids=["interleaved", "planar"])
    @pytest.mark.parametrize("ome", [True, False])
    def test_rgb_samples_are_rejected_before_channel_names(
        self, tmp_path: Path, mocker: MockerFixture, *, planar: bool, ome: bool
    ) -> None:
        """Steinbock images are scientific channel stacks; a samples axis is never read as colours."""
        sample, _, _ = _write_sample(tmp_path)
        shape = (3, 48, 64) if planar else (48, 64, 3)
        tifffile.imwrite(
            sample.image,
            np.zeros(shape, dtype=np.uint8),
            ome=ome,
            photometric="rgb",
            planarconfig="separate" if planar else "contig",
        )
        names = mocker.spy(steinbock_images, "ome_channel_names")

        with pytest.raises(ReaderFormatError, match="Axis 's' has length 3") as caught:
            read_rasters(sample, make_raster_options(), metadata=None, calibrated_frame=sample.name)

        assert caught.value.context.artifact == "OME image"
        assert caught.value.path == sample.image
        names.assert_not_called()

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

        result = read_rasters(sample, make_raster_options(), metadata=None, calibrated_frame=sample.name)

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

        result = read_rasters(sample, make_raster_options(), metadata=None, calibrated_frame=sample.name)

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
            result = read_rasters(
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
        assert result.grid.frame == "sample_pixels"
        assert result.grid.calibration is None

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

        result = read_rasters(sample, make_raster_options(chunks=chunks), metadata=None, calibrated_frame=sample.name)

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

        result = read_rasters(sample, make_raster_options(factors), metadata=None, calibrated_frame=sample.name)

        assert isinstance(result.image, DataTree)
        assert isinstance(result.labels, DataTree)
        assert len(result.labels.children) == len(factors) + 1
        for level in result.labels.children.values():
            values = level["image"].data.compute()
            assert values.dtype == np.dtype(dtype)
            np.testing.assert_array_equal(np.unique(values), identifiers)
            np.testing.assert_array_equal(values[np.ix_([0, -1], [0, -1])], identifiers.reshape(2, 2))


class TestCalibration:
    """Prefer images.csv extents, fall back to complete image OME sizes and require agreement."""

    _SHAPE = (8, 12)

    def _metadata(self, tmp_path: Path, name: str, sizes: Mapping[str, float | str]) -> TiffMetadata:
        path = tmp_path / f"{name}.ome.tiff"
        tifffile.imwrite(path, np.zeros(self._SHAPE, dtype=np.uint16), ome=True, metadata={"axes": "YX", **sizes})
        return inspect_tiff(ArtifactFile(path), context=_CONTEXT)

    def test_images_csv_extents_take_precedence(self, tmp_path: Path) -> None:
        image = self._metadata(tmp_path, "image", {})
        labels = self._metadata(tmp_path, "labels", {})

        calibration = _calibration(image, labels, ImageMetadata(self._SHAPE, (0.5, 2.5)), self._SHAPE)

        assert calibration is not None
        assert calibration.spacing_xy_um == (0.5, 2.5)
        assert calibration.source == (
            "images.csv acquisition_width_um/width_px and acquisition_height_um/height_px; "
            "resolved XY=(0.5, 2.5) micrometer/pixel"
        )

    def test_complete_image_ome_sizes_calibrate(self, tmp_path: Path) -> None:
        image = self._metadata(
            tmp_path, "image", {"PhysicalSizeX": 300, "PhysicalSizeXUnit": "nm", "PhysicalSizeY": 0.7}
        )
        labels = self._metadata(tmp_path, "labels", {"PhysicalSizeY": 0.7})

        calibration = _calibration(image, labels, ImageMetadata(self._SHAPE, None), self._SHAPE)

        assert calibration is not None
        assert calibration.spacing_xy_um == (0.3, 0.7)
        assert calibration.source.startswith("Source image OME PhysicalSizeX/PhysicalSizeY")

    @pytest.mark.parametrize("row", [None, ImageMetadata(_SHAPE, None)])
    def test_partial_or_absent_declarations_keep_pixels(self, tmp_path: Path, row: ImageMetadata | None) -> None:
        image = self._metadata(tmp_path, "image", {"PhysicalSizeX": 0.5})
        labels = self._metadata(tmp_path, "labels", {"PhysicalSizeX": 0.5, "PhysicalSizeY": 2.0})

        assert _calibration(image, labels, row, self._SHAPE) is None

    def test_last_bit_differences_agree(self, tmp_path: Path) -> None:
        """An exact 808.917 µm / 1829 px and a float division of the same values differ in the last bit."""
        exact, divided = 0.44227282668124657, 808.917 / 1829
        assert exact != divided
        image = self._metadata(tmp_path, "image", {"PhysicalSizeX": divided, "PhysicalSizeY": 1.0})
        labels = self._metadata(tmp_path, "labels", {"PhysicalSizeX": divided})

        calibration = _calibration(image, labels, ImageMetadata(self._SHAPE, (exact, 1.0)), self._SHAPE)

        assert calibration is not None
        assert calibration.spacing_xy_um == (exact, 1.0)

    @pytest.mark.parametrize(
        ("csv", "artifact"),
        [(True, "image"), (True, "labels"), (False, "labels")],
        ids=["csv-image", "csv-mask", "ome-mask"],
    )
    def test_disagreeing_declaration_fails(self, tmp_path: Path, *, csv: bool, artifact: str) -> None:
        """Without images.csv the image's own sizes are the reference the mask must match."""
        conflicting = {"PhysicalSizeX": 0.5 * (1 + 1e-9)}
        image = self._metadata(tmp_path, "image", conflicting if artifact == "image" else {"PhysicalSizeX": 0.5})
        labels = self._metadata(tmp_path, "labels", conflicting if artifact == "labels" else {})
        row = ImageMetadata(self._SHAPE, (0.5, 2.5)) if csv else None

        with pytest.raises(ReaderFormatError, match="X spacing") as caught:
            _calibration(image, labels, row, self._SHAPE)

        assert caught.value.path == (image if artifact == "image" else labels).path

    def test_stale_images_csv_dimensions_fail(self, tmp_path: Path) -> None:
        image = self._metadata(tmp_path, "image", {})
        labels = self._metadata(tmp_path, "labels", {})

        with pytest.raises(ReaderFormatError, match=r"images\.csv dimensions \(8, 11\)"):
            _calibration(image, labels, ImageMetadata((8, 11), (0.5, 2.5)), self._SHAPE)


class TestReadRastersFrames:
    """Calibrated samples use the micrometre frame; uncalibrated ones a separate pixel frame."""

    @pytest.mark.parametrize(
        ("metadata", "frame"),
        [(None, "sample_pixels"), (ImageMetadata((48, 64), (0.5, 2.0)), "global")],
        ids=["uncalibrated", "calibrated"],
    )
    def test_frame_follows_calibration(self, tmp_path: Path, metadata: ImageMetadata | None, frame: str) -> None:
        sample, _, _ = _write_sample(tmp_path)

        result = read_rasters(sample, make_raster_options(), metadata=metadata, calibrated_frame="global")

        assert result.grid.frame == frame
        for element in (result.image, result.labels):
            transforms = get_transformation(element, get_all=True)
            assert isinstance(transforms, dict)
            assert list(transforms) == [frame]
