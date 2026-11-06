"""Read the DBiT barcode map into one sequence-to-ordinal lookup per grid axis.

DBiT-seq delivers barcodes through two orthogonal sets of microfluidic channels
(https://github.com/MingyuYang-Yale/DBiT-seq). Each spot of the resulting
square grid carries one barcode from set A and one from set B. The barcode map
lists each set separately, one position and its eight-base sequence per row::

    A1    AACCGGTT
    A2    TTGGCCAA
    B1    GGGGTTTT
    B2    CCCCAAAA

A combined per-spot map (one 16-base pair per grid location, as published with
the original E10 data) is not this format; split it into its two axes first.

The schema is documented in ``_schemas``. The file is small by construction:
at most fifty positions per axis, so at most 100 rows are parsed and one more
is read to detect a longer file.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import pandas as pd

from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers.dbit._schemas import BARCODE_SOURCE_DTYPE, MAX_GRID_SIDE, normalize_barcodes

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialdata_io.readers._utils.paths import ArtifactFile

_CONTEXT = ReaderErrorContext("dbit", "barcode positions")


@dataclass(frozen=True, slots=True)
class Barcodes:
    """Immutable upper-case sequence-to-ordinal lookups for the A and B axes."""

    a: Mapping[str, int]
    b: Mapping[str, int]


def read_barcodes(path: ArtifactFile) -> Barcodes:
    """Read and validate the barcode map."""
    try:
        # The Python parser keeps NUL bytes, so the schema rejects them instead
        # of a C parser silently truncating the field.
        with path.open(encoding="utf-8-sig", newline="") as source:
            frame = pd.read_csv(
                source,
                sep="\t",
                header=None,
                dtype=BARCODE_SOURCE_DTYPE,
                keep_default_na=False,
                engine="python",
                nrows=2 * MAX_GRID_SIDE + 1,
            )
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        add_reader_context(error, context=_CONTEXT, path=path)
        raise

    normalize_barcodes(frame, path)

    a, b = (
        MappingProxyType(dict(zip(frame.loc[selected, "sequence"], frame.loc[selected, "position"], strict=True)))
        for selected in (frame["axis"].eq("A"), frame["axis"].eq("B"))
    )
    return Barcodes(a, b)
