"""Column contracts of Xenium cells, boundaries and transcript tables.

Parquet files and their CSV twins share one header per artifact. Named columns may
appear in any order; documented columns are typed explicitly, and duplicate or
empty names are rejected before any parser could repair them. Cell identifiers
are integers in the integer-ID family (XOA 1.0-1.2) and text otherwise; binary
text (XOA 1.0 feature names, XOA 1.3 cell IDs) is decoded strictly as UTF-8.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from spatialdata_io.readers.xenium._version import Family

STRING = pa.string()
STRING_DTYPE = pd.ArrowDtype(STRING)
FLOAT64 = pa.float64()
FLOAT32 = pa.float32()
INT64 = pa.int64()

CELL_ID = "cell_id"
CELL_FLOATS = ("x_centroid", "y_centroid", "cell_area", "nucleus_area")
CELL_REQUIRED_COUNTS = ("transcript_counts", "control_probe_counts", "control_codeword_counts", "total_counts")
CELL_OPTIONAL_COUNTS = (
    "unassigned_codeword_counts",
    "deprecated_codeword_counts",
    "genomic_control_counts",
    "nucleus_count",
)
CELL_REQUIRED = (CELL_ID, *CELL_FLOATS, *CELL_REQUIRED_COUNTS)

VERTICES = ("vertex_x", "vertex_y")
LABEL_ID = "label_id"

# Documented transcript columns; later builds add the optional ones. Unknown
# columns are not projected, so a new format adds its columns here explicitly.
TRANSCRIPT_REQUIRED: dict[str, pa.DataType] = {
    "transcript_id": pa.uint64(),
    CELL_ID: STRING,
    "overlaps_nucleus": pa.uint8(),
    "feature_name": STRING,
    "x_location": FLOAT32,
    "y_location": FLOAT32,
    "z_location": FLOAT32,
    "qv": FLOAT32,
}
TRANSCRIPT_OPTIONAL: dict[str, pa.DataType] = {
    "fov_name": STRING,
    "nucleus_distance": FLOAT32,
    "codeword_index": pa.int32(),
    "codeword_category": STRING,
    "is_gene": pa.bool_(),
}


def schema_error(path: Path, artifact: str, *, found: str, expected: str) -> ReaderFormatError:
    """Describe one Xenium artifact contract violation."""
    return ReaderFormatError(context=ReaderErrorContext("xenium", artifact), path=path, found=found, expected=expected)


def validate_header(names: Sequence[str], required: Sequence[str], *, path: Path, artifact: str) -> None:
    """Reject empty, duplicate or missing required column names."""
    if any(not name for name in names) or len(set(names)) != len(names):
        raise schema_error(path, artifact, found=f"columns {list(names)!r}", expected="unique nonempty column names")
    missing = [name for name in required if name not in names]
    if missing:
        raise schema_error(path, artifact, found=f"missing columns {missing!r}", expected=f"columns {list(required)!r}")


def id_type(family: Family) -> pa.DataType:
    """Return the identifier type: integers before XOA 1.3, text afterwards."""
    return INT64 if family == "integer_ids" else STRING


def cell_types(names: Sequence[str], family: Family) -> dict[str, pa.DataType]:
    """Return explicit Arrow types for a cells header; unknown columns are kept as text."""
    types: dict[str, pa.DataType] = dict.fromkeys(names, STRING)
    types[CELL_ID] = id_type(family)
    types |= dict.fromkeys(CELL_FLOATS, FLOAT64)
    types |= {name: INT64 for name in (*CELL_REQUIRED_COUNTS, *CELL_OPTIONAL_COUNTS) if name in names}
    return types


def transcript_types(names: Sequence[str], *, path: Path) -> dict[str, pa.DataType]:
    """Return the projected documented transcript columns and their output types."""
    validate_header(names, tuple(TRANSCRIPT_REQUIRED), path=path, artifact="transcripts")
    return TRANSCRIPT_REQUIRED | {name: kind for name, kind in TRANSCRIPT_OPTIONAL.items() if name in names}


def cast_column(column: pa.ChunkedArray, kind: pa.DataType, name: str, *, path: Path, artifact: str) -> pa.ChunkedArray:
    """Cast a projected column to its documented type; lossy casts raise."""
    if column.type == kind:
        return column
    try:
        return pc.cast(column, kind)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
        raise schema_error(
            path, artifact, found=f"{name} of type {column.type}", expected=f"{name} as {kind}"
        ) from error


def normalize_text(column: pa.ChunkedArray, name: str, *, path: Path, artifact: str) -> pa.ChunkedArray:
    """Return non-null ``string`` values; binary text is decoded strictly as UTF-8."""
    kind = column.type
    if not (
        pa.types.is_string(kind)
        or pa.types.is_large_string(kind)
        or pa.types.is_binary(kind)
        or pa.types.is_large_binary(kind)
    ):
        raise schema_error(path, artifact, found=f"{name} of type {kind}", expected=f"{name} as UTF-8 text")
    try:
        # Arrow validates UTF-8 when casting binary to text.
        values = column if kind == STRING else pc.cast(column, STRING)
    except pa.ArrowInvalid as error:
        raise schema_error(
            path, artifact, found=f"{name} that is not UTF-8", expected=f"{name} as UTF-8 text"
        ) from error
    if values.null_count:
        raise schema_error(path, artifact, found=f"missing {name} values", expected=f"complete {name} values")
    return values


def normalize_ids(column: pa.ChunkedArray, family: Family, *, path: Path, artifact: str) -> pa.ChunkedArray:
    """Return non-null int64 identifiers (integer family) or UTF-8 text identifiers."""
    if family != "integer_ids":
        return normalize_text(column, CELL_ID, path=path, artifact=artifact)
    if not pa.types.is_integer(column.type) or column.null_count:
        raise schema_error(
            path, artifact, found=f"identifiers of type {column.type}", expected="non-null integer identifiers"
        )
    return cast_column(column, INT64, CELL_ID, path=path, artifact=artifact)


def text_series(column: pa.Array | pa.ChunkedArray) -> pd.Series:
    """View Arrow text as an Arrow-backed pandas series without copying it."""
    return pd.Series(pd.arrays.ArrowExtensionArray(column), dtype=STRING_DTYPE)


def id_index(column: pa.Array | pa.ChunkedArray) -> pd.Index:
    """Return normalized int64 or text identifiers as an index for exact lookups."""
    if pa.types.is_integer(column.type):
        return pd.Index(np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.int64))
    return pd.Index(text_series(column), dtype="str")


def integer_ids(text: pd.Index, *, path: Path, artifact: str) -> pd.Index:
    """Parse canonical decimal identifiers (XOA 1.0-1.2 barcodes) to int64 exactly."""
    try:
        values = pd.Index(text.to_numpy(dtype=np.str_).astype(np.int64))
    except (ValueError, OverflowError) as error:
        raise schema_error(path, artifact, found="non-integer cell IDs", expected="decimal integer cell IDs") from error
    if not values.astype("str").equals(text.astype("str")):
        raise schema_error(
            path, artifact, found="non-canonical integer cell IDs", expected="canonical decimal cell IDs"
        )
    return values


def vocabulary_codes(
    values: pd.Series, vocabulary: pd.Index, name: str, *, path: Path, artifact: str
) -> NDArray[np.intp]:
    """Return the vocabulary position of every value; unknown values raise.

    Factorizing hashes the values once (an Arrow kernel for Arrow-backed text);
    only the distinct values are looked up in the vocabulary.
    """
    codes, values_uniques = values.factorize()
    uniques = pd.Index(values_uniques)
    positions = vocabulary.get_indexer(uniques)
    unknown = positions < 0
    if unknown.any():
        examples = uniques[unknown][:5].tolist()
        raise schema_error(
            path, artifact, found=f"unknown {name} values {examples!r}", expected=f"{name} values from the count matrix"
        )
    return positions[codes]


def require_finite(
    values: NDArray[np.floating], name: str, *, path: Path, artifact: str, allow_nan: bool = False
) -> None:
    """Reject infinite values, and NaN unless the column documents it."""
    invalid = np.isinf(values) if allow_nan else ~np.isfinite(values)
    if invalid.any():
        raise schema_error(path, artifact, found=f"nonfinite {name} values", expected=f"finite {name} values")


def require_no_nulls(table: pa.Table, *, path: Path, artifact: str) -> None:
    """Reject nulls; Xenium writes none in the projected columns."""
    missing = [name for name in table.column_names if table.column(name).null_count]
    if missing:
        raise schema_error(path, artifact, found=f"missing values in {missing!r}", expected="complete source records")
