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

The schema is documented in ``_schemas``; at most 100 records describe fifty
positions per axis. No expression or image data passes through this parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow.csv as pacsv

from spatialdata_io.readers._utils.csv import csv_errors
from spatialdata_io.readers._utils.errors import add_reader_context
from spatialdata_io.readers.dbit._schemas import BARCODE_CONTEXT, BARCODE_TYPES, normalize_barcodes

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialdata_io.readers._utils.paths import ArtifactFile


@dataclass(frozen=True, slots=True)
class Barcodes:
    """Immutable upper-case sequence-to-ordinal lookups for the A and B axes."""

    a: Mapping[str, int]
    b: Mapping[str, int]


def read_barcodes(path: ArtifactFile) -> Barcodes:
    """Read and validate the barcode map."""
    try:
        with path.open("rb") as source, csv_errors(context=BARCODE_CONTEXT, path=path):
            frame = pacsv.read_csv(
                source,
                read_options=pacsv.ReadOptions(column_names=BARCODE_TYPES.names, use_threads=False),
                parse_options=pacsv.ParseOptions(delimiter="\t", newlines_in_values=True),
                convert_options=pacsv.ConvertOptions(
                    column_types=BARCODE_TYPES,
                    include_columns=BARCODE_TYPES.names,
                    null_values=[],
                    strings_can_be_null=False,
                ),
            ).to_pandas(types_mapper=pd.ArrowDtype)
    except OSError as error:
        add_reader_context(error, context=BARCODE_CONTEXT, path=path)
        raise
    normalize_barcodes(frame, path)

    a, b = (
        MappingProxyType(dict(zip(frame.loc[selected, "sequence"], frame.loc[selected, "position"], strict=True)))
        for selected in (frame["axis"].eq("A"), frame["axis"].eq("B"))
    )
    return Barcodes(a, b)
