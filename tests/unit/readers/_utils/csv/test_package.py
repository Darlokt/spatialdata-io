"""Tests for the narrow CSV package surface."""

from __future__ import annotations

from spatialdata_io.readers._utils import csv


class TestCsvPackage:
    def test_exports_only_header_errors_and_lazy_blocks(self) -> None:
        assert csv.__all__ == [
            "csv_errors",
            "csv_header",
            "read_csv_blocks",
        ]
