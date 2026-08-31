"""Integration tests for native TIFF selection graphs and task resources."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import cloudpickle  # type: ignore[import-untyped]  # third-party package has no type information
import dask
import numpy as np
import pytest
import tifffile
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


class TestReadTiff:
    """Read explicit series and levels through serializable native-chunk tasks."""

    @pytest.mark.parametrize(
        "storage",
        [
            (None, None, 5, ((1, 1, 1), (5, 5, 5, 4), (23,))),
            (None, (16, 16), None, ((1, 1, 1), (16, 3), (16, 7))),
            ("deflate", (16, 16), None, ((1, 1, 1), (16, 3), (16, 7))),
            ("lzw", (16, 16), None, ((1, 1, 1), (16, 3), (16, 7))),
        ],
    )
    def test_reads_native_chunks_exactly(
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
        "options",
        [
            TiffOptions(compression="deflate", tile=(16, 16)),
            TiffOptions(compression="deflate", rowsperstrip=5),
        ],
    )
    def test_small_selection_reads_only_one_native_segment(
        self,
        cyx_data: NDArray[np.uint16],
        write_tiff: TiffFactory,
        mocker: MockerFixture,
        options: TiffOptions,
    ) -> None:
        path = write_tiff(cyx_data, options)
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
        assert read_payloads == {0}
        assert len(closed_handles) == 1
        assert closed_handles[0].closed
        assert len(closed_stores) == 1

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
