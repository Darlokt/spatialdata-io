"""CSV header reading, Arrow error context and lazy CSV blocks for readers.

Readers declare Arrow types for their columns and call :mod:`pyarrow.csv`
directly; Arrow's typed CSV grammar is their contract. :func:`csv_header` reads
the header with Arrow and rejects duplicate names, :func:`csv_errors` reports
malformed content with reader context, and :func:`read_csv_blocks` partitions
an uncompressed CSV lazily into typed Arrow-backed Dask partitions.
"""

from __future__ import annotations

from spatialdata_io.readers._utils.csv._arrow import csv_errors, csv_header
from spatialdata_io.readers._utils.csv._blocks import read_csv_blocks

__all__ = [
    "csv_errors",
    "csv_header",
    "read_csv_blocks",
]
