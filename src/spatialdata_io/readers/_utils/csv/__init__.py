"""Sequential CSV streams, Arrow header and error handling, and lazy CSV blocks.

Readers declare Arrow types for their columns and call :mod:`pyarrow.csv`
directly; Arrow's typed CSV grammar is their contract. :func:`csv_header` reads
the header with Arrow and rejects duplicate names, :func:`csv_errors` reports
malformed content with reader context, and :func:`read_csv_blocks` partitions
an uncompressed CSV lazily into typed Arrow-backed Dask partitions.
:func:`open_csv_stream` owns a plain or gzip stream for sequential reads.
:func:`csv_data_buffer` checks physical CSV lines and protects a literal initial
U+FEFF before Arrow parses a data partition.
"""

from __future__ import annotations

from spatialdata_io.readers._utils.csv._arrow import csv_errors, csv_header
from spatialdata_io.readers._utils.csv._blocks import csv_data_buffer, read_csv_blocks
from spatialdata_io.readers._utils.csv._stream import open_csv_stream

__all__ = [
    "csv_data_buffer",
    "csv_errors",
    "csv_header",
    "open_csv_stream",
    "read_csv_blocks",
]
