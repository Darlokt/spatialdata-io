"""Tests for single-tile TIFF selections and tile extent validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import imagecodecs
import numpy as np
import pytest
import tifffile

from spatialdata_io.readers._utils.errors import RasterFormatError
from spatialdata_io.readers._utils.image.tiff._tiles import read_single_tile

if TYPE_CHECKING:
    from pathlib import Path


class TestReadSingleTile:
    def test_reads_a_predictor_encoded_big_endian_corner_tile(self, tmp_path: Path) -> None:
        path = tmp_path / "predictor.tif"
        source = (np.arange(19 * 23, dtype=np.uint16) * 101).reshape(19, 23)
        tifffile.imwrite(
            path,
            source,
            tile=(16, 16),
            byteorder=">",
            compression="deflate",
            predictor=True,
            extratags=[(274, "H", 1, 6, False)],
        )
        selection = (slice(17, 19), slice(17, 23))

        with tifffile.TiffFile(path) as tiff:
            result = read_single_tile(tiff, tiff.pages.first, source.shape, selection)

        assert result is not None
        assert result.dtype == source.dtype
        np.testing.assert_array_equal(result, source[selection])

    def test_reads_interleaved_jpeg_when_decoder_initialization_seeks(self, tmp_path: Path) -> None:
        path = tmp_path / "jpeg.tif"
        source = np.arange(19 * 23 * 3, dtype=np.uint8).reshape(19, 23, 3)
        tifffile.imwrite(path, source, tile=(16, 16), compression="jpeg", photometric="rgb")
        selection = (slice(16, 19), slice(16, 23), slice(1, 3))
        expected = tifffile.imread(path)[selection]

        with tifffile.TiffFile(path) as tiff:
            result = read_single_tile(tiff, tiff.pages.first, source.shape, selection)

        assert result is not None
        np.testing.assert_array_equal(result, expected)

    def test_reads_a_tile_encoded_without_edge_padding(self, tmp_path: Path) -> None:
        path = tmp_path / "unpadded.tif"
        source = np.arange(19 * 23, dtype=np.uint8).reshape(19, 23)
        segments = (
            bytes(imagecodecs.deflate_encode(source[y : y + 16, x : x + 16].tobytes()))
            for y in (0, 16)
            for x in (0, 16)
        )
        tifffile.imwrite(path, segments, shape=source.shape, dtype=source.dtype, tile=(16, 16), compression="deflate")
        selection = (slice(16, 19), slice(16, 23))

        with tifffile.TiffFile(path) as tiff:
            result = read_single_tile(tiff, tiff.pages.first, source.shape, selection)

        assert result is not None
        np.testing.assert_array_equal(result, source[selection])

    def test_selects_the_plane_in_series_order(self, tmp_path: Path) -> None:
        path = tmp_path / "stack.ome.tif"
        source = np.arange(2 * 3 * 19 * 23, dtype=np.uint16).reshape(2, 3, 19, 23)
        tifffile.imwrite(
            path, source, tile=(16, 16), photometric="minisblack", metadata={"axes": "TCYX"}, compression="deflate"
        )
        selection = (slice(1, 2), slice(2, 3), slice(16, 19), slice(16, 23))

        with tifffile.TiffFile(path) as tiff:
            result = read_single_tile(tiff, tiff.series[0], source.shape, selection)

        assert result is not None
        np.testing.assert_array_equal(result, source[selection])

    def test_defers_selections_spanning_multiple_tiles(self, tmp_path: Path) -> None:
        path = tmp_path / "multi-tile.tif"
        source = np.ones((19, 23), dtype=np.uint8)
        tifffile.imwrite(path, source, tile=(16, 16))

        with tifffile.TiffFile(path) as tiff:
            assert read_single_tile(tiff, tiff.pages.first, source.shape, (slice(15, 17), slice(0, 2))) is None

    def test_rejects_a_tile_with_a_byte_count_beyond_eof(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt-tile.tif"
        source = np.ones((19, 23), dtype=np.uint8)
        tifffile.imwrite(path, source, tile=(16, 16), compression="deflate")
        with tifffile.TiffFile(path, mode="r+") as tiff:
            page = tiff.pages.first
            page.tags["TileByteCounts"].overwrite((path.stat().st_size, *page.databytecounts[1:]))

        with (
            tifffile.TiffFile(path) as tiff,
            pytest.raises(RasterFormatError, match="TIFF tile 0 of IFD 0 extends past the end of the file"),
        ):
            read_single_tile(tiff, tiff.pages.first, source.shape, (slice(0, 16), slice(0, 16)))
