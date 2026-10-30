"""Validate the bounded DBiT barcode-position map."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import pandas as pd

from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers.dbit._schemas import BARCODE_SOURCE_DTYPE, _normalize_barcodes

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialdata_io.readers._utils.paths import ArtifactFile


@dataclass(frozen=True, slots=True)
class _Barcodes:
    """Own immutable nucleotide-to-position lookups for the two grid axes."""

    a: Mapping[str, int]
    b: Mapping[str, int]


def _read_barcodes(path: ArtifactFile) -> _Barcodes:
    """Read at most fifty unique eight-base barcodes per contiguous grid axis."""
    context = ReaderErrorContext("dbit", "barcode positions")
    try:
        # One extra row proves violation of the protocol's fifty positions per axis.
        # The Python parser preserves NUL bytes for the semantic schema to reject.
        with path.open(encoding="utf-8-sig", newline="") as source:
            frame = pd.read_csv(
                source,
                sep="\t",
                header=None,
                dtype=BARCODE_SOURCE_DTYPE,
                keep_default_na=False,
                engine="python",
                nrows=101,
            )
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
