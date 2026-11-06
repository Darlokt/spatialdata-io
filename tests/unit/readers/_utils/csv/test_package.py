"""Tests for the narrow CSV package surface."""

from __future__ import annotations

from spatialdata_io.readers._utils import csv


class TestCsvPackage:
    def test_exports_only_header_and_block_readers(self) -> None:
        assert csv.__all__ == ["parse_csv_header", "read_csv_blocks", "read_csv_header"]
