"""Integration tests for native TIFF selection graphs and task resources."""

from __future__ import annotations

from dataclasses import replace
from math import prod
from typing import TYPE_CHECKING, Literal, NamedTuple

import cloudpickle
import dask
import dask.array as da
import numpy as np
import pytest
import tifffile
import zarr
from tifffile.zarr import ZarrTiffStore

import spatialdata_io.readers._utils.image.tiff._dask as tiff_dask
from spatialdata_io.readers._utils.errors import RasterFormatError, ReaderErrorContext
from spatialdata_io.readers._utils.image import inspect_tiff, read_tiff
from spatialdata_io.readers._utils.image.tiff._dask import _read_tiff_selection, _TiffArray
from spatialdata_io.readers._utils.paths import ArtifactFile
from tests.integration.readers._utils.image.conftest import TiffOptions

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from pytest_mock import MockerFixture

    from tests.integration.readers._utils.image.conftest import TiffFactory


_RASTER_CONTEXT = ReaderErrorContext("test", "raster")


class _FinalPage(NamedTuple):
    """Storage of one uncompressed contiguous single-page TIFF."""

    dtype: str
    byteorder: Literal["<", ">"]
    photometric: str
    shape: tuple[int, ...]


class TestReadTiff:
    """Read explicit series and levels through serializable native-chunk tasks."""

    @pytest.mark.parametrize("storage", ["single", "stack", "shaped-axes", "virtual-imagej", "imagej"])
    def test_single_uncompressed_strips_split_into_bounded_bands(
        self, tmp_path: Path, mocker: MockerFixture, storage: str
    ) -> None:
        path = ArtifactFile(tmp_path / "single-strip.tif")
        plane_shape = (15, 9)
        shape = plane_shape if storage == "single" else (3, *plane_shape)
        source = (np.arange(prod(shape), dtype=np.uint16) * 3 + 1).reshape(shape)
        tifffile.imwrite(
            path,
            source,
            photometric="minisblack",
            imagej=storage.endswith("imagej"),
            truncate=storage == "virtual-imagej",
            metadata={"axes": "YXC"} if storage == "shaped-axes" else {},
            rowsperstrip=plane_shape[0],
            byteorder=">",
        )
        row_items = plane_shape[1]
        mocker.patch.object(tiff_dask, "STRIP_BAND_BYTES", 4 * row_items * source.itemsize)
        with tifffile.TiffFile(path) as tiff:
            assert tiff.pages.first.databytecounts == (prod(plane_shape) * source.itemsize,)
            assert len(tiff.pages) == (1 if storage in {"single", "virtual-imagej"} else 3)
        metadata = inspect_tiff(path, context=_RASTER_CONTEXT)
        read_array = mocker.spy(tifffile.FileHandle, "read_array")
        zarr_open = mocker.spy(tiff_dask.zarr, "open_array")
        raster = read_tiff(metadata, series=0, level=0, context=_RASTER_CONTEXT)
        if storage == "shaped-axes":
            assert raster.axes == ("Y", "X", "C")
        result = raster.data
        read_array.assert_not_called()
        assert result.chunks[-2] == (4, 4, 4, 3)

        crop = (slice(5, 7), slice(2, 5))
        if storage != "single":
            crop = (slice(1, 2), *crop)
        np.testing.assert_array_equal(result[crop].compute(scheduler="synchronous"), source[crop])
        assert [call.kwargs["count"] for call in read_array.call_args_list] == [4 * row_items]
        zarr_open.assert_not_called()
        np.testing.assert_array_equal(result.compute(scheduler="synchronous"), source)

    def test_physical_page_uses_its_own_axes_dtype_and_tiles(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = ArtifactFile(tmp_path / "pages.qptiff")
        source = np.arange(32 * 35 * 3, dtype=np.uint8).reshape(32, 35, 3)
        late_plane = np.arange(7 * 9, dtype=np.float32).reshape(7, 9)
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.ones((7, 9), dtype=np.float32), software="PerkinElmer-QPI test fixture", metadata=None)
            writer.write(source, tile=(16, 16), photometric="rgb", compression="deflate", metadata=None)
            writer.write(late_plane, metadata=None)
        metadata = inspect_tiff(path, context=_RASTER_CONTEXT, include_pages=True)
        selection_spy = mocker.spy(tiff_dask, "_read_tiff_selection")
        result = read_tiff(metadata, page=1, context=_RASTER_CONTEXT)
        selection_spy.assert_not_called()
        assert result.axes == ("Y", "X", "S")
        assert result.color == "rgb"
        assert result.data.dtype == source.dtype
        assert result.data.chunks == ((16, 16), (16, 16, 3), (3,))
        restored = cloudpickle.loads(cloudpickle.dumps(result.data))
        np.testing.assert_array_equal(restored[17:20, 18:24].compute(scheduler="synchronous"), source[17:20, 18:24])
        selection_spy.assert_called_once()
        np.testing.assert_array_equal(result.data.compute(), source)
        np.testing.assert_array_equal(read_tiff(metadata, page=2, context=_RASTER_CONTEXT).data.compute(), late_plane)

    def test_physical_page_splits_uncompressed_strips(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = ArtifactFile(tmp_path / "pages.tif")
        source = np.arange(15 * 9, dtype=np.float32).reshape(15, 9)
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.ones((7, 9), dtype=np.uint16), metadata=None)
            writer.write(source, rowsperstrip=15, metadata=None)
        metadata = inspect_tiff(path, context=_RASTER_CONTEXT, include_pages=True)
        mocker.patch.object(tiff_dask, "STRIP_BAND_BYTES", 4 * 9 * source.itemsize)
        read_array = mocker.spy(tifffile.FileHandle, "read_array")
        result = read_tiff(metadata, page=1, context=_RASTER_CONTEXT).data

        read_array.assert_not_called()
        assert result.chunks == ((4, 4, 4, 3), (9,))
        np.testing.assert_array_equal(result[13:, 1:3].compute(scheduler="synchronous"), source[13:, 1:3])
        assert [call.kwargs["count"] for call in read_array.call_args_list] == [3 * 9]

    @pytest.mark.parametrize(
        ("selection", "error", "message"),
        [
            ((-1, None, None), IndexError, "page index"),
            ((1, None, None), IndexError, "page index"),
            ((0, 0, None), ValueError, "cannot be combined"),
            ((0, None, 0), ValueError, "cannot be combined"),
            ((None, 0, None), ValueError, "both series and level"),
        ],
    )
    def test_requires_one_inspected_selection(
        self,
        tmp_path: Path,
        selection: tuple[int | None, int | None, int | None],
        error: type[ValueError | IndexError],
        message: str,
    ) -> None:
        path = ArtifactFile(tmp_path / "page.tif")
        tifffile.imwrite(path, np.ones((8, 8), dtype=np.uint16))
        metadata = inspect_tiff(path, context=_RASTER_CONTEXT, include_pages=True)
        page, series, level = selection

        with pytest.raises(error, match=message) as caught:
            read_tiff(metadata, page=page, series=series, level=level, context=_RASTER_CONTEXT)

        assert caught.value.__notes__ == [f"While handling {_RASTER_CONTEXT.describe(path=path)}."]

    def test_requires_page_inspection(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "page.tif")
        tifffile.imwrite(path, np.ones((8, 8), dtype=np.uint16))
        metadata = inspect_tiff(path, context=_RASTER_CONTEXT)

        with pytest.raises(ValueError, match="include_pages=True"):
            read_tiff(metadata, page=0, context=_RASTER_CONTEXT)

    def test_missing_linked_plane_at_compute(self, linked_ome_tiff: tuple[Path, NDArray[np.uint16]]) -> None:
        path, _ = linked_ome_tiff
        image = read_tiff(
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT), series=0, level=0, context=_RASTER_CONTEXT
        )
        path.with_name("second.ome.tif").unlink()
        with pytest.raises(RasterFormatError, match=r"missing.*plane"):
            image.data[1].compute(scheduler="synchronous")

    def test_incomplete_plane_mapping_at_inspection(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "incomplete.ome.tif")
        xml = """<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
        <Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16"
        SizeX="5" SizeY="4" SizeZ="1" SizeC="2" SizeT="1"><Channel ID="Channel:0"/>
        <Channel ID="Channel:1"/><TiffData IFD="0" PlaneCount="1"/></Pixels></Image></OME>"""
        tifffile.imwrite(path, np.ones((4, 5), dtype=np.uint16), description=xml, metadata=None)
        with pytest.raises(RasterFormatError, match=r"missing.*plane"):
            inspect_tiff(path, context=_RASTER_CONTEXT)

    @pytest.mark.parametrize("dtype", ["int64", "uint64", ">i8", ">u8"])
    def test_64bit_dtype_is_interoperable_without_pixel_conversion(self, tmp_path: Path, dtype: str) -> None:
        path = tmp_path / "labels.tif"
        source_dtype = np.dtype(dtype)
        source = np.asarray([[0, 2**53 + 1], [3, np.iinfo(source_dtype).max]], dtype=source_dtype)
        tifffile.imwrite(path, source, photometric="minisblack")

        result = read_tiff(
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT), series=0, level=0, context=_RASTER_CONTEXT
        )

        # TIFF decoding already returns native byte order, including for files
        # whose encoded pixels are big-endian.
        assert result.data.dtype == source.dtype.newbyteorder("=")
        target = tmp_path / "labels.zarr"
        da.to_zarr(result.data, target)
        np.testing.assert_array_equal(zarr.open_array(target, mode="r")[:], source)

    @pytest.mark.parametrize("compression", [None, "deflate"])
    @pytest.mark.parametrize("shape", [(1, 32, 40), (1, 1, 40), (1, 32, 1)])
    def test_shaped_singleton_channels_preserve_spatial_dimensions(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        compression: str | None,
        shape: tuple[int, int, int],
    ) -> None:
        path = tmp_path / "singleton.tif"
        source = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        tifffile.imwrite(
            path,
            source,
            photometric="minisblack",
            planarconfig="separate",
            metadata={"axes": "CYX"},
            compression=compression,
            tile=(16, 16),
        )
        selection_spy = mocker.spy(tiff_dask, "_read_tiff_selection")

        result = read_tiff(
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT), series=0, level=0, context=_RASTER_CONTEXT
        )

        selection_spy.assert_not_called()
        assert result.axes == ("Y", "X")
        assert result.data.shape == shape[1:]
        assert result.data.dtype == source.dtype
        restored = cloudpickle.loads(cloudpickle.dumps(result.data))
        np.testing.assert_array_equal(restored[:2, :3].compute(scheduler="synchronous"), source[0, :2, :3])
        selection_spy.assert_called_once()
        np.testing.assert_array_equal(result.data.compute(), source[0])

    @pytest.mark.parametrize(
        "storage",
        [
            # Strips of whole rows are grouped into one bounded band per plane.
            (None, None, 5, ((1, 1, 1), (19,), (23,))),
            ("deflate", None, 1, ((1, 1, 1), (19,), (23,))),
            (None, (16, 16), None, ((1, 1, 1), (16, 3), (16, 7))),
            ("deflate", (16, 16), None, ((1, 1, 1), (16, 3), (16, 7))),
            ("lzw", (16, 16), None, ((1, 1, 1), (16, 3), (16, 7))),
        ],
    )
    def test_reads_storage_aligned_chunks_exactly(
        self,
        cyx_data: NDArray[np.uint16],
        write_tiff: TiffFactory,
        storage: tuple[
            str | None,
            tuple[int, int] | None,
            int | None,
            tuple[tuple[int, ...], ...],
        ],
    ) -> None:
        compression, tile, rowsperstrip, expected_chunks = storage
        path = write_tiff(
            cyx_data,
            TiffOptions(compression=compression, tile=tile, rowsperstrip=rowsperstrip),
        )

        result = read_tiff(inspect_tiff(path, context=_RASTER_CONTEXT), series=0, level=0, context=_RASTER_CONTEXT)

        assert result.data.chunks == expected_chunks
        assert result.axes == ("C", "Y", "X")
        np.testing.assert_array_equal(result.data.compute(), cyx_data)

    def test_selection_task_reads_exact_requested_region(
        self,
        cyx_data: NDArray[np.uint16],
        write_tiff: TiffFactory,
    ) -> None:
        path = write_tiff(cyx_data, TiffOptions(compression="deflate", tile=(16, 16)))
        metadata = inspect_tiff(path, context=_RASTER_CONTEXT)
        level = metadata.series[0].levels[0]
        spec = _TiffArray(metadata.path, (0, 0), level.shape, level.dtype, level.direct_row_axis is not None)

        result = _read_tiff_selection(
            spec,
            (slice(1, 2), slice(3, 9), slice(4, 11)),
        )

        np.testing.assert_array_equal(result, cyx_data[1:2, 3:9, 4:11])

    @pytest.mark.parametrize(
        ("options", "expected_payloads"),
        [
            (TiffOptions(compression="deflate", tile=(16, 16)), {0}),
            # Bands of two 5-row strips: the selection reads exactly its band.
            (TiffOptions(compression="deflate", rowsperstrip=5), {0, 1}),
        ],
    )
    def test_small_selection_reads_only_its_tile_or_row_band(
        self,
        cyx_data: NDArray[np.uint16],
        write_tiff: TiffFactory,
        mocker: MockerFixture,
        options: TiffOptions,
        expected_payloads: set[int],
    ) -> None:
        path = write_tiff(cyx_data, options)
        mocker.patch.object(tiff_dask, "STRIP_BAND_BYTES", 2 * 5 * 23 * cyx_data.itemsize)
        result = read_tiff(inspect_tiff(path, context=_RASTER_CONTEXT), series=0, level=0, context=_RASTER_CONTEXT)
        with tifffile.TiffFile(path) as tiff:
            payload_ranges = tuple(
                (offset, offset + byte_count)
                for page in tiff.pages
                for offset, byte_count in zip(page.dataoffsets, page.databytecounts, strict=True)
            )

        reads: list[tuple[int, int]] = []
        original_read = tifffile.FileHandle.read
        original_file_close = tifffile.FileHandle.close
        original_store_close = ZarrTiffStore.close
        closed_handles: list[tifffile.FileHandle] = []
        closed_stores: list[ZarrTiffStore] = []

        def tracked_read(handle: tifffile.FileHandle, size: int = -1, /) -> bytes:
            start = handle.tell()
            value = original_read(handle, size)
            reads.append((start, start + len(value)))
            return value

        def tracked_file_close(handle: tifffile.FileHandle) -> None:
            original_file_close(handle)
            closed_handles.append(handle)

        def tracked_store_close(store: ZarrTiffStore) -> None:
            original_store_close(store)
            closed_stores.append(store)

        mocker.patch.object(tifffile.FileHandle, "read", new=tracked_read)
        mocker.patch.object(tifffile.FileHandle, "close", new=tracked_file_close)
        mocker.patch.object(ZarrTiffStore, "close", new=tracked_store_close)

        np.testing.assert_array_equal(
            result.data[:1, :2, :3].compute(scheduler="synchronous"),
            cyx_data[:1, :2, :3],
        )

        read_payloads = {
            index
            for index, (payload_start, payload_stop) in enumerate(payload_ranges)
            if any(max(payload_start, read_start) < min(payload_stop, read_stop) for read_start, read_stop in reads)
        }
        assert read_payloads == expected_payloads
        assert len(closed_handles) == 1
        assert closed_handles[0].closed
        assert len(closed_stores) == int(options.tile is None)

    @pytest.mark.parametrize("compression", [None, "zlib"])
    def test_one_row_strips_form_strip_aligned_bands(
        self, tmp_path: Path, mocker: MockerFixture, compression: str | None
    ) -> None:
        path = tmp_path / "strips.tif"
        source = np.arange(2 * 11 * 7, dtype=np.uint16).reshape(2, 11, 7)
        tifffile.imwrite(
            path, source, photometric="minisblack", metadata={"axes": "CYX"}, rowsperstrip=1, compression=compression
        )
        # Three one-row strips per band leave a final band of two rows.
        mocker.patch.object(tiff_dask, "STRIP_BAND_BYTES", 3 * 7 * source.itemsize + 1)

        result = read_tiff(
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT), series=0, level=0, context=_RASTER_CONTEXT
        )

        assert result.data.chunks == ((1, 1), (3, 3, 3, 2), (7,))
        np.testing.assert_array_equal(result.data.compute(), source)
        np.testing.assert_array_equal(result.data[1, 4:8, 2:5].compute(), source[1, 4:8, 2:5])

    @pytest.mark.parametrize(
        "storage",
        [
            _FinalPage("uint16", "<", "minisblack", (11, 7)),
            _FinalPage("uint16", ">", "minisblack", (11, 7)),
            _FinalPage("float32", "<", "minisblack", (11, 7)),
            _FinalPage("uint8", "<", "rgb", (11, 7, 3)),
        ],
        ids=lambda storage: f"{storage.dtype}-{storage.byteorder}-{storage.photometric}",
    )
    def test_final_pages_are_read_directly_in_bounded_bands(
        self, tmp_path: Path, mocker: MockerFixture, storage: _FinalPage
    ) -> None:
        path = tmp_path / "final.tif"
        source = (np.arange(np.prod(storage.shape)) * 3 + 1).astype(storage.dtype).reshape(storage.shape)
        tifffile.imwrite(path, source, photometric=storage.photometric, rowsperstrip=1, byteorder=storage.byteorder)
        row_items = int(np.prod(storage.shape[1:]))
        mocker.patch.object(tiff_dask, "STRIP_BAND_BYTES", 4 * row_items * source.itemsize)
        metadata = inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)
        zarr_open = mocker.spy(tiff_dask.zarr, "open_array")
        read_array = mocker.spy(tifffile.FileHandle, "read_array")
        file_close = mocker.spy(tifffile.FileHandle, "close")

        result = read_tiff(metadata, series=0, level=0, context=_RASTER_CONTEXT)

        assert result.data.chunks[0] == (4, 4, 3)
        np.testing.assert_array_equal(result.data[5:7].compute(scheduler="synchronous"), source[5:7])
        # One read of exactly the second band, without the Zarr strip decoder.
        assert [call.kwargs["count"] for call in read_array.call_args_list] == [4 * row_items]
        assert file_close.call_count == 1
        np.testing.assert_array_equal(result.data.compute(), source)
        assert result.data.dtype == source.dtype.newbyteorder("=")
        zarr_open.assert_not_called()

    @pytest.mark.parametrize(
        ("data", "encoding"),
        [
            pytest.param(np.arange(3 * 9 * 5, dtype=np.uint16).reshape(3, 9, 5), (None, False), id="multi-page"),
            pytest.param(np.arange(9 * 5, dtype=np.uint16).reshape(9, 5), ("zlib", False), id="compressed"),
            pytest.param(np.arange(9 * 5, dtype=np.uint16).reshape(9, 5), ("zlib", True), id="predictor"),
        ],
    )
    def test_other_storage_uses_the_strip_decoder(
        self, tmp_path: Path, mocker: MockerFixture, data: NDArray[np.uint16], encoding: tuple[str | None, bool]
    ) -> None:
        path = tmp_path / "general.tif"
        compression, predictor = encoding
        tifffile.imwrite(
            path, data, photometric="minisblack", rowsperstrip=1, compression=compression, predictor=predictor
        )
        metadata = inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)
        zarr_open = mocker.spy(tiff_dask.zarr, "open_array")

        result = read_tiff(metadata, series=0, level=0, context=_RASTER_CONTEXT)

        np.testing.assert_array_equal(result.data.compute(scheduler="synchronous"), data)
        assert zarr_open.call_count == prod(map(len, result.data.chunks))

    def test_partial_width_selection_of_a_final_page_uses_the_strip_decoder(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        path = tmp_path / "final.tif"
        source = np.arange(9 * 5, dtype=np.uint16).reshape(9, 5)
        tifffile.imwrite(path, source, photometric="minisblack", rowsperstrip=1)
        metadata = inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)
        level = metadata.series[0].levels[0]
        spec = _TiffArray(metadata.path, (0, 0), level.shape, level.dtype, level.direct_row_axis is not None)
        zarr_open = mocker.spy(tiff_dask.zarr, "open_array")

        result = _read_tiff_selection(spec, (slice(2, 6), slice(1, 4)))

        np.testing.assert_array_equal(result, source[2:6, 1:4])
        zarr_open.assert_called_once()

    def test_single_page_series_in_a_linked_file_is_read_from_that_file(self, tmp_path: Path) -> None:
        # Image 1 of the OME document in first.ome.tif is stored in second.ome.tif.
        first_uuid = "urn:uuid:44444444-4444-4444-8444-444444444444"
        second_uuid = "urn:uuid:55555555-5555-4555-8555-555555555555"

        def image(index: int, uuid: str, file_name: str) -> str:
            return (
                f'<Image ID="Image:{index}"><Pixels ID="Pixels:{index}" DimensionOrder="XYZCT" Type="uint16" '
                'SizeX="5" SizeY="4" SizeZ="1" SizeC="1" SizeT="1">'
                f'<Channel ID="Channel:{index}:0" SamplesPerPixel="1"/><TiffData IFD="0" PlaneCount="1">'
                f'<UUID FileName="{file_name}">{uuid}</UUID></TiffData></Pixels></Image>'
            )

        namespace = 'xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"'
        first = np.arange(20, dtype=np.uint16).reshape(4, 5)
        second = first + 100
        xml = (
            f'<OME {namespace} UUID="{first_uuid}">'
            f"{image(0, first_uuid, 'first.ome.tif')}{image(1, second_uuid, 'second.ome.tif')}</OME>"
        )
        tifffile.imwrite(tmp_path / "first.ome.tif", first, description=xml, metadata=None, rowsperstrip=1)
        binary_only = (
            f'<OME {namespace} UUID="{second_uuid}">'
            f'<BinaryOnly MetadataFile="first.ome.tif" UUID="{first_uuid}"/></OME>'
        )
        tifffile.imwrite(tmp_path / "second.ome.tif", second, description=binary_only, metadata=None, rowsperstrip=1)

        result = read_tiff(
            inspect_tiff(ArtifactFile(tmp_path / "first.ome.tif"), context=_RASTER_CONTEXT),
            series=1,
            level=0,
            context=_RASTER_CONTEXT,
        )

        np.testing.assert_array_equal(result.data.compute(), second)

    def test_empty_strip_is_not_read_as_pixels(self, tmp_path: Path) -> None:
        # Offset and byte count zero mark a strip without data; the decoder fills it.
        path = tmp_path / "sparse.tif"
        tifffile.imwrite(path, np.full((16, 8), 7, dtype=np.uint16))
        with tifffile.TiffFile(path, mode="r+") as tiff:
            tiff.pages.first.tags["StripOffsets"].overwrite(0)
            tiff.pages.first.tags["StripByteCounts"].overwrite(0)

        result = read_tiff(
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT), series=0, level=0, context=_RASTER_CONTEXT
        )

        np.testing.assert_array_equal(result.data.compute(), np.zeros((16, 8), dtype=np.uint16))

    def test_empty_tile_preserves_fill_and_adjacent_pixels(self, tmp_path: Path) -> None:
        path = tmp_path / "sparse-tile.tif"
        source = np.full((16, 25), 7, dtype=np.uint16)
        tifffile.imwrite(path, source, tile=(16, 16), compression="deflate", extratags=[(42113, "s", 0, "17", False)])
        with tifffile.TiffFile(path, mode="r+") as tiff:
            page = tiff.pages.first
            page.tags["TileOffsets"].overwrite((0, page.dataoffsets[1]))
            page.tags["TileByteCounts"].overwrite((0, page.databytecounts[1]))

        result = read_tiff(
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT), series=0, level=0, context=_RASTER_CONTEXT
        )

        source[:, :16] = 17
        np.testing.assert_array_equal(result.data.compute(scheduler="synchronous"), source)

    def test_truncated_strip_fails_instead_of_reading_past_it(self, tmp_path: Path) -> None:
        path = tmp_path / "truncated.tif"
        tifffile.imwrite(path, np.full((16, 8), 7, dtype=np.uint16))
        with tifffile.TiffFile(path, mode="r+") as tiff:
            page = tiff.pages.first
            page.tags["StripByteCounts"].overwrite(page.nbytes - 32)

        result = read_tiff(
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT), series=0, level=0, context=_RASTER_CONTEXT
        )

        with pytest.raises(tifffile.TiffFileError, match="corrupted strip"):
            result.data.compute()

    def test_contiguous_rgb_strips_form_bands(self, tmp_path: Path) -> None:
        path = tmp_path / "rgb.tif"
        source = np.arange(9 * 5 * 3, dtype=np.uint8).reshape(9, 5, 3)
        tifffile.imwrite(path, source, photometric="rgb", rowsperstrip=2)

        result = read_tiff(
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT), series=0, level=0, context=_RASTER_CONTEXT
        )

        assert result.axes == ("Y", "X", "S")
        assert result.color == "rgb"
        assert result.data.chunks == ((9,), (5,), (3,))
        np.testing.assert_array_equal(result.data.compute(), source)

    def test_reads_jpeg_compressed_tiles(self, tmp_path: Path) -> None:
        path = tmp_path / "jpeg-compressed.tif"
        source = np.arange(3 * 32 * 32, dtype=np.uint8).reshape(3, 32, 32)
        tifffile.imwrite(
            path,
            source,
            compression="jpeg",
            tile=(16, 16),
            photometric="rgb",
            planarconfig="separate",
            metadata={"axes": "CYX"},
        )

        result = read_tiff(
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT), series=0, level=0, context=_RASTER_CONTEXT
        )

        assert tuple(axis_chunks[0] for axis_chunks in result.data.chunks) == (1, 16, 16)
        np.testing.assert_allclose(result.data.compute(), source, atol=5)

    def test_selects_bigtiff_multiseries_and_pyramid_levels(self, tmp_path: Path) -> None:
        series_path = tmp_path / "series.btf"
        first = np.arange(35, dtype=np.uint8).reshape(5, 7)
        second = np.arange(99, dtype=np.uint16).reshape(9, 11)
        with tifffile.TiffWriter(series_path, bigtiff=True) as writer:
            writer.write(first, photometric="minisblack", metadata={"axes": "YX"})
            writer.write(second, photometric="minisblack", metadata={"axes": "YX"})

        series_metadata = inspect_tiff(ArtifactFile(series_path), context=_RASTER_CONTEXT)
        np.testing.assert_array_equal(
            read_tiff(series_metadata, series=1, level=0, context=_RASTER_CONTEXT).data.compute(),
            second,
        )

        pyramid_path = tmp_path / "pyramid.tif"
        base = np.arange(38 * 46, dtype=np.uint16).reshape(38, 46)
        reduced = base[::2, ::2]
        with tifffile.TiffWriter(pyramid_path) as writer:
            writer.write(
                base,
                tile=(16, 16),
                subifds=1,
                photometric="minisblack",
                compression="deflate",
                metadata={"axes": "YX"},
            )
            writer.write(reduced, tile=(16, 16), subfiletype=1, photometric="minisblack", compression="deflate")

        pyramid_metadata = inspect_tiff(ArtifactFile(pyramid_path), context=_RASTER_CONTEXT)
        result = read_tiff(pyramid_metadata, series=0, level=1, context=_RASTER_CONTEXT)
        assert result.data.chunks == ((16, 3), (16, 7))
        np.testing.assert_array_equal(result.data.compute(), reduced)

    def test_reads_linked_ome_planes(self, linked_ome_tiff: tuple[Path, NDArray[np.uint16]]) -> None:
        path, expected = linked_ome_tiff

        result = read_tiff(
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT), series=0, level=0, context=_RASTER_CONTEXT
        )

        assert result.data.chunks == ((1, 1), (4,), (5,))
        np.testing.assert_array_equal(result.data.compute(), expected)

    def test_construction_rechunk_and_small_selection_are_lazy_and_culled(
        self,
        cyx_data: NDArray[np.uint16],
        write_tiff: TiffFactory,
        mocker: MockerFixture,
    ) -> None:
        path = write_tiff(cyx_data, TiffOptions(compression="deflate", tile=(16, 16)))
        selection_spy = mocker.spy(tiff_dask, "_read_tiff_selection")

        result = read_tiff(inspect_tiff(path, context=_RASTER_CONTEXT), series=0, level=0, context=_RASTER_CONTEXT)
        rechunked = result.data.rechunk((1, 8, 8))

        assert selection_spy.call_count == 0
        np.testing.assert_array_equal(
            rechunked[:1, :2, :3].compute(scheduler="synchronous"),
            cyx_data[:1, :2, :3],
        )
        assert selection_spy.call_count == 1

    def test_graph_is_deterministic_path_only_serializable_and_process_safe(
        self,
        cyx_data: NDArray[np.uint16],
        write_tiff: TiffFactory,
    ) -> None:
        path = write_tiff(cyx_data, TiffOptions(compression="deflate", tile=(16, 16)))
        metadata = inspect_tiff(path, context=_RASTER_CONTEXT)
        first = read_tiff(metadata, series=0, level=0, context=_RASTER_CONTEXT).data
        second = read_tiff(metadata, series=0, level=0, context=_RASTER_CONTEXT).data

        assert first.name == second.name
        assert len(first.dask.layers) == 2
        assert {type(layer).__name__ for layer in first.dask.layers.values()} == {
            "Blockwise",
            "MaterializedLayer",
        }
        restored = cloudpickle.loads(cloudpickle.dumps(first))
        graph_text = repr(first.dask)
        assert "TiffFile" not in graph_text
        assert "ZarrTiffStore" not in graph_text
        assert "FileHandle" not in graph_text
        with dask.config.set(scheduler="processes"):
            np.testing.assert_array_equal(restored.compute(), cyx_data)

    def test_optimized_small_slice_culls_unrelated_source_tasks(
        self,
        cyx_data: NDArray[np.uint16],
        write_tiff: TiffFactory,
    ) -> None:
        path = write_tiff(cyx_data, TiffOptions(compression="deflate", tile=(16, 16)))
        data = read_tiff(inspect_tiff(path, context=_RASTER_CONTEXT), series=0, level=0, context=_RASTER_CONTEXT).data
        selection = data[:1, :2, :3]

        optimized = dask.optimize(selection)[0]

        assert len(optimized.dask) == 2

    @pytest.mark.parametrize(
        ("series", "level", "match"),
        [
            (-1, 0, "series index"),
            (1, 0, "series index"),
            (0, -1, "level index"),
            (0, 1, "level index"),
        ],
    )
    def test_rejects_invalid_indices_with_reader_context(
        self,
        cyx_data: NDArray[np.uint16],
        write_tiff: TiffFactory,
        series: int,
        level: int,
        match: str,
    ) -> None:
        path = write_tiff(cyx_data)
        metadata = inspect_tiff(path, context=_RASTER_CONTEXT)
        context = ReaderErrorContext("reader", "stitched image", detected_version="2.0")

        with pytest.raises(IndexError, match=match) as caught:
            read_tiff(metadata, series=series, level=level, context=context)

        assert caught.value.__notes__ == [f"While handling {context.describe(path=path)}."]
        assert caught.value.__notes__[0] == (
            f"While handling reader 'reader', artifact 'stitched image', detected version '2.0', path {str(path)!r}."
        )

    def test_closes_resources_and_adds_selection_context_after_source_change(
        self,
        cyx_data: NDArray[np.uint16],
        write_tiff: TiffFactory,
    ) -> None:
        path = write_tiff(cyx_data, TiffOptions(compression="deflate", tile=(16, 16)))
        result = read_tiff(inspect_tiff(path, context=_RASTER_CONTEXT), series=0, level=0, context=_RASTER_CONTEXT)
        path.write_bytes(b"corrupt")

        with pytest.raises(tifffile.TiffFileError, match="not a TIFF") as caught:
            result.data[:1, :16, :16].compute(scheduler="synchronous")
        assert any("series=0" in note and "selection=" in note for note in caught.value.__notes__)

    def test_validates_task_result_against_inspected_dtype(
        self,
        cyx_data: NDArray[np.uint16],
        write_tiff: TiffFactory,
        mocker: MockerFixture,
    ) -> None:
        path = write_tiff(cyx_data, TiffOptions(compression="deflate", tile=(16, 16)))
        metadata = inspect_tiff(path, context=_RASTER_CONTEXT)
        series = metadata.series[0]
        wrong_level = replace(series.levels[0], dtype=np.dtype(np.uint8))
        inconsistent = replace(metadata, series=(replace(series, levels=(wrong_level,)),))
        store_close = mocker.spy(ZarrTiffStore, "close")
        file_close = mocker.spy(tifffile.FileHandle, "close")

        result = read_tiff(inconsistent, series=0, level=0, context=_RASTER_CONTEXT)

        with pytest.raises(RasterFormatError, match="does not match inspected metadata"):
            result.data[:1, :16, :16].compute(scheduler="synchronous")
        store_close.assert_not_called()
        assert file_close.call_count >= 1
