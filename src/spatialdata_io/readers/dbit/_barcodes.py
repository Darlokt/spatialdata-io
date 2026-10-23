"""Validate the bounded DBiT barcode-position map."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from types import MappingProxyType
from typing import TYPE_CHECKING

import pandas as pd

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialdata_io.readers._utils.paths import ArtifactFile

# A complete 100-row map needs about 1.3 KiB. Leave room for text formatting
# while bounding allocation for a mistakenly selected or malformed file.
_MAX_BARCODE_BYTES = 16_384
_MAX_GRID_SIDE = 50
_BARCODE_COLUMNS = 2


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

        frame = pd.read_csv(BytesIO(contents), sep="\t", header=None, dtype=str, keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        add_reader_context(error, context=context, path=path)
        raise

    if frame.shape[1] != _BARCODE_COLUMNS or not 0 < len(frame) <= 2 * _MAX_GRID_SIDE:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"barcode map shape {frame.shape}",
            expected="a nonempty headerless two-column TSV with at most fifty A and fifty B positions",
        )

    positions = frame.iloc[:, 0]
    sequences = frame.iloc[:, 1].str.upper()
    if (
        not positions.str.fullmatch(r"[AB]([1-9]|[1-4][0-9]|50)").all()
        or not sequences.str.fullmatch(r"[ACGT]{8}").all()
    ):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="invalid barcode position or sequence",
            expected="positions A1..A50 or B1..B50 and eight A/C/G/T bases (case-insensitive)",
        )
    if not positions.is_unique:
        raise ReaderFormatError(
            context=context, path=path, found="duplicate positions", expected="one barcode per A/B position"
        )

    # Sequence uniqueness is axis-local: the same DNA can identify one A and
    # one B position because its place in the combined observation ID differs.
    maps: list[Mapping[str, int]] = []
    for axis in ("A", "B"):
        selected = positions.str.startswith(axis)
        axis_sequences = sequences[selected]
        numbers = positions[selected].str[1:].astype(int)
        if not axis_sequences.is_unique or set(numbers) != set(range(1, len(numbers) + 1)):
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"duplicate barcodes or noncontiguous {axis} positions",
                expected="unique sequences within each axis and contiguous positions starting at one",
            )

        maps.append(MappingProxyType(dict(zip(axis_sequences, numbers, strict=True))))

    # Equal contiguous axes make their length the side of the square grid.
    if not maps[0] or len(maps[0]) != len(maps[1]):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="unequal or missing grid axes",
            expected="the same nonempty A and B extent",
        )

    return _Barcodes(maps[0], maps[1])
