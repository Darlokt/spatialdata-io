"""Strict CSV header parsing and lazy newline-block CSV partitions for readers.

Readers keep their column contracts, dtypes, semantic normalization,
coordinate meaning and SpatialData model construction.
"""

from __future__ import annotations

from spatialdata_io.readers._utils.csv._blocks import read_csv_blocks
from spatialdata_io.readers._utils.csv._header import parse_csv_header, read_csv_header

__all__ = ["parse_csv_header", "read_csv_blocks", "read_csv_header"]
