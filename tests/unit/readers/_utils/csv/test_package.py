"""Tests for the narrow CSV package surface."""

from __future__ import annotations

from spatialdata_io.readers._utils import csv


class TestCsvPackage:
    def test_exports_header_record_and_number_readers(self) -> None:
        assert csv.__all__ == [
            "LITERAL_ZERO",
            "csv_errors",
            "csv_header",
            "decode_float64",
            "decode_float64_block",
            "decode_integer_block",
            "decode_integers",
            "iter_csv_batches",
            "parse_csv_header",
            "read_csv_blocks",
            "read_csv_header",
            "read_csv_table",
        ]
