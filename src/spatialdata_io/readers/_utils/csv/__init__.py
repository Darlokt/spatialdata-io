"""CSV header reading, Arrow error context and lazy CSV blocks for readers.

Readers declare Arrow types for their columns and call :mod:`pyarrow.csv`
directly; Arrow's typed CSV grammar is their contract. :func:`csv_header` reads
the header with Arrow and rejects duplicate names, :func:`csv_errors` reports
malformed content with reader context, and :func:`read_csv_blocks` partitions
an uncompressed CSV lazily into typed Arrow-backed Dask partitions.

The text-record readers (:func:`iter_csv_batches`, :func:`read_csv_table`),
the strict stdlib header parsers (:func:`parse_csv_header`,
:func:`read_csv_header`) and the exact text decoders (``decode_*``,
``LITERAL_ZERO``) serve readers that still parse numbers from text.
"""

from __future__ import annotations

from spatialdata_io.readers._utils.csv._arrow import csv_errors, csv_header
from spatialdata_io.readers._utils.csv._batches import iter_csv_batches, read_csv_table
from spatialdata_io.readers._utils.csv._blocks import read_csv_blocks
from spatialdata_io.readers._utils.csv._header import parse_csv_header, read_csv_header
from spatialdata_io.readers._utils.csv._numbers import (
    LITERAL_ZERO,
    decode_float64,
    decode_float64_block,
    decode_integer_block,
    decode_integers,
)

__all__ = [
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
