"""Integration tests for native TIFF selection graphs and task resources."""

from __future__ import annotations

from dataclasses import replace
from math import prod
from typing import TYPE_CHECKING, Literal, NamedTuple

import cloudpickle  # type: ignore[import-untyped]  # third-party package has no type information
import dask
import dask.array as da
import numpy as np
import pytest
import tifffile
import zarr
from tifffile.zarr import ZarrTiffStore

import spatialdata_io.readers._utils.image.tiff._dask as tiff_dask
from spatialdata_io.readers._utils.errors import RasterFormatError
from spatialdata_io.readers._utils.image import inspect_tiff, read_tiff
from spatialdata_io.readers._utils.image.tiff._dask import _read_tiff_selection
from spatialdata_io.readers._utils.image.tiff._types import _TiffReadSpec
from tests.integration.readers._utils.image.conftest import TiffOptions

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from pytest_mock import MockerFixture

    from tests.integration.readers._utils.image.conftest import TiffFactory


class _FinalPage(NamedTuple):
    """Storage of one uncompressed contiguous single-page TIFF."""

    dtype: str
    byteorder: Literal["<", ">"]
    photometric: str
    shape: tuple[int, ...]


class TestReadTiff:
    """Read explicit series and levels through serializable native-chunk tasks."""

    @pytest.mark.parametrize("dtype", ["int64", "uint64", ">i8", ">u8"])
    def test_64bit_dtype_is_interoperable_without_pixel_conversion(self, tmp_path: Path, dtype: str) -> None:
        path = tmp_path / "labels.tif"
        source_dtype = np.dtype(dtype)
        source = np.asarray([[0, 2**53 + 1], [3, np.iinfo(source_dtype).max]], dtype=source_dtype)
        tifffile.imwrite(path, source, photometric="minisblack")

        result = read_tiff(inspect_tiff(path), series=0, level=0)

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

        result = read_tiff(inspect_tiff(path), series=0, level=0)

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

        result = read_tiff(inspect_tiff(path), series=0, level=0)

        assert result.data.chunks == expected_chunks
        assert result.axes == ("C", "Y", "X")
        np.testing.assert_array_equal(result.data.compute(), cyx_data)

    def test_selection_task_reads_exact_requested_region(
        self,
        cyx_data: NDArray[np.uint16],
        write_tiff: TiffFactory,
    ) -> None:
        path = write_tiff(cyx_data, TiffOptions(compression="deflate", tile=(16, 16)))
        metadata = inspect_tiff(path)
        spec = _TiffReadSpec(
            path=metadata.path,
            series=0,
            level=0,
            dtype=metadata.series[0].levels[0].dtype,
        )

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
        result = read_tiff(inspect_tiff(path), series=0, level=0)
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
        assert len(closed_stores) == 1

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

        result = read_tiff(inspect_tiff(path), series=0, level=0)

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
        metadata = inspect_tiff(path)
        zarr_open = mocker.spy(tiff_dask.zarr, "open_array")
        read_array = mocker.spy(tifffile.FileHandle, "read_array")
        file_close = mocker.spy(tifffile.FileHandle, "close")

        result = read_tiff(metadata, series=0, level=0)

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
        metadata = inspect_tiff(path)
        zarr_open = mocker.spy(tiff_dask.zarr, "open_array")

        result = read_tiff(metadata, series=0, level=0)

        np.testing.assert_array_equal(result.data.compute(scheduler="synchronous"), data)
        assert zarr_open.call_count == prod(map(len, result.data.chunks))

    def test_partial_width_selection_of_a_final_page_uses_the_strip_decoder(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        path = tmp_path / "final.tif"
        source = np.arange(9 * 5, dtype=np.uint16).reshape(9, 5)
        tifffile.imwrite(path, source, photometric="minisblack", rowsperstrip=1)
        metadata = inspect_tiff(path)
        spec = _TiffReadSpec(path=metadata.path, series=0, level=0, dtype=metadata.series[0].levels[0].dtype)
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

        result = read_tiff(inspect_tiff(tmp_path / "first.ome.tif"), series=1, level=0)

        np.testing.assert_array_equal(result.data.compute(), second)

    def test_empty_strip_is_not_read_as_pixels(self, tmp_path: Path) -> None:
        # Offset and byte count zero mark a strip without data; the decoder fills it.
        path = tmp_path / "sparse.tif"
        tifffile.imwrite(path, np.full((16, 8), 7, dtype=np.uint16))
        with tifffile.TiffFile(path, mode="r+") as tiff:
            tiff.pages.first.tags["StripOffsets"].overwrite(0)
            tiff.pages.first.tags["StripByteCounts"].overwrite(0)

        result = read_tiff(inspect_tiff(path), series=0, level=0)

        np.testing.assert_array_equal(result.data.compute(), np.zeros((16, 8), dtype=np.uint16))

    def test_truncated_strip_fails_instead_of_reading_past_it(self, tmp_path: Path) -> None:
        path = tmp_path / "truncated.tif"
        tifffile.imwrite(path, np.full((16, 8), 7, dtype=np.uint16))
        with tifffile.TiffFile(path, mode="r+") as tiff:
            page = tiff.pages.first
            page.tags["StripByteCounts"].overwrite(page.nbytes - 32)

        result = read_tiff(inspect_tiff(path), series=0, level=0)

        with pytest.raises(tifffile.TiffFileError, match="corrupted strip"):
            result.data.compute()

    def test_contiguous_rgb_strips_form_bands(self, tmp_path: Path) -> None:
        path = tmp_path / "rgb.tif"
        source = np.arange(9 * 5 * 3, dtype=np.uint8).reshape(9, 5, 3)
        tifffile.imwrite(path, source, photometric="rgb", rowsperstrip=2)

        result = read_tiff(inspect_tiff(path), series=0, level=0)

        assert result.axes == ("Y", "X", "S")
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

        result = read_tiff(inspect_tiff(path), series=0, level=0)

        assert tuple(axis_chunks[0] for axis_chunks in result.data.chunks) == (1, 16, 16)
        np.testing.assert_allclose(result.data.compute(), source, atol=5)

    def test_selects_bigtiff_multiseries_and_pyramid_levels(self, tmp_path: Path) -> None:
        series_path = tmp_path / "series.btf"
        first = np.arange(35, dtype=np.uint8).reshape(5, 7)
        second = np.arange(99, dtype=np.uint16).reshape(9, 11)
        with tifffile.TiffWriter(series_path, bigtiff=True) as writer:
            writer.write(first, photometric="minisblack", metadata={"axes": "YX"})
            writer.write(second, photometric="minisblack", metadata={"axes": "YX"})

        series_metadata = inspect_tiff(series_path)
        np.testing.assert_array_equal(
            read_tiff(series_metadata, series=1, level=0).data.compute(),
            second,
        )

        pyramid_path = tmp_path / "pyramid.tif"
        base = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
        reduced = base[::2, ::2]
        with tifffile.TiffWriter(pyramid_path) as writer:
            writer.write(
                base,
                tile=(16, 16),
                subifds=1,
                photometric="minisblack",
                metadata={"axes": "YX"},
            )
            writer.write(reduced, tile=(16, 16), subfiletype=1, photometric="minisblack")

        pyramid_metadata = inspect_tiff(pyramid_path)
        np.testing.assert_array_equal(
            read_tiff(pyramid_metadata, series=0, level=1).data.compute(),
            reduced,
        )

    def test_reads_linked_ome_planes(self, linked_ome_tiff: tuple[Path, NDArray[np.uint16]]) -> None:
        path, expected = linked_ome_tiff

        result = read_tiff(inspect_tiff(path), series=0, level=0)

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

        result = read_tiff(inspect_tiff(path), series=0, level=0)
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
        metadata = inspect_tiff(path)
        first = read_tiff(metadata, series=0, level=0).data
        second = read_tiff(metadata, series=0, level=0).data

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
        data = read_tiff(inspect_tiff(path), series=0, level=0).data
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
    def test_rejects_invalid_indices(
        self,
        cyx_data: NDArray[np.uint16],
        write_tiff: TiffFactory,
        series: int,
        level: int,
        match: str,
    ) -> None:
        metadata = inspect_tiff(write_tiff(cyx_data))

        with pytest.raises(IndexError, match=match):
            read_tiff(metadata, series=series, level=level)

    def test_closes_resources_and_adds_selection_context_after_source_change(
        self,
        cyx_data: NDArray[np.uint16],
        write_tiff: TiffFactory,
    ) -> None:
        path = write_tiff(cyx_data, TiffOptions(compression="deflate", tile=(16, 16)))
        result = read_tiff(inspect_tiff(path), series=0, level=0)
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
        metadata = inspect_tiff(path)
        series = metadata.series[0]
        wrong_level = replace(series.levels[0], dtype=np.dtype(np.uint8))
        inconsistent = replace(metadata, series=(replace(series, levels=(wrong_level,)),))
        store_close = mocker.spy(ZarrTiffStore, "close")
        file_close = mocker.spy(tifffile.FileHandle, "close")

        result = read_tiff(inconsistent, series=0, level=0)

        with pytest.raises(RasterFormatError, match="does not match inspected metadata"):
            result.data[:1, :16, :16].compute(scheduler="synchronous")
        assert store_close.call_count >= 1
        assert file_close.call_count >= 1
