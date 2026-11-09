"""Strict CSV headers, Arrow text records and exact number decoding for readers.

Readers parse the header strictly before any parser can repair names, read the
records as text, eagerly in record batches or as one table from a stream they
own, or lazily in newline-aligned blocks, and decode numeric columns from their
text without silent precision loss. Readers keep their column contracts,
missing-value spellings, semantic normalization, coordinate meaning and
SpatialData model construction.
"""

from __future__ import annotations

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
