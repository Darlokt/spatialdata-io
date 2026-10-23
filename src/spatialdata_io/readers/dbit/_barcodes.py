"""Validate the bounded DBiT barcode-position map."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from types import MappingProxyType
from typing import TYPE_CHECKING

import pandas as pd

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers.dbit._schemas import BARCODE_SOURCE_DTYPE, _normalize_barcodes

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialdata_io.readers._utils.paths import ArtifactFile

# A complete 100-row map needs about 1.3 KiB. Leave room for text formatting
# while bounding allocation for a mistakenly selected or malformed file.
_MAX_BARCODE_BYTES = 16_384


@dataclass(frozen=True, slots=True)
class _Barcodes:
    """Own immutable nucleotide-to-position lookups for the two grid axes."""

    a: Mapping[str, int]
    b: Mapping[str, int]


def _read_barcodes(path: ArtifactFile) -> _Barcodes:
    """Read at most fifty unique eight-base barcodes per contiguous grid axis."""
    context = ReaderErrorContext("dbit", "barcode positions")
    try:
        # The protocol has at most 100 rows. Bound malformed files before pandas
        # allocates a frame. The file owner also closes on parser failure.
        with path.open("rb") as source:
            contents = source.read(_MAX_BARCODE_BYTES + 1)

        if len(contents) > _MAX_BARCODE_BYTES:
            raise ReaderFormatError(
                context=context, path=path, found="oversized barcode map", expected="at most 16 KiB of barcode metadata"
            )

        # Pandas' C parser silently truncates fields at NUL bytes.
        if b"\x00" in contents:
            raise ReaderFormatError(
                context=context, path=path, found="NUL bytes in barcode map", expected="text fields without NUL bytes"
            )

        frame = pd.read_csv(BytesIO(contents), sep="\t", header=None, dtype=BARCODE_SOURCE_DTYPE, keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        add_reader_context(error, context=context, path=path)
        raise

    _normalize_barcodes(frame, path)

    maps: list[Mapping[str, int]] = []
    for axis in ("A", "B"):
        selected = frame["axis"].eq(axis)
        maps.append(
            MappingProxyType(dict(zip(frame.loc[selected, "sequence"], frame.loc[selected, "position"], strict=True)))
        )

    return _Barcodes(maps[0], maps[1])
