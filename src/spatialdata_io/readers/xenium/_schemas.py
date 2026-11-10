"""Arrow types and column contracts of Xenium cells, boundaries, transcripts and analysis.

Parquet files and their CSV twins share one header per artifact. Named columns may
appear in any order; documented columns have declared Arrow types per bundle
family, and duplicate or empty names are rejected before any parser could repair
them. Cells/boundary identifiers are positive ``uint64`` in the integer-ID family
(XOA 1.0-1.2); transcript assignments stay signed ``int64`` to retain ``-1``. Other
families use text IDs; binary text (XOA 1.0 feature names, XOA 1.3 cell IDs) is decoded
strictly as UTF-8. Parquet columns are cast to the declared types; CSV twins are
parsed by Arrow directly into them, so numbers follow Arrow's CSV grammar. The
exception are signed transcript cell IDs, which a CSV must spell as decimal
digits or ``-1``, because Arrow wraps hexadecimal in a signed type.
Native cell summaries require float64 storage with unique nonempty string column
names and the required nucleus centroid columns. Secondary analysis instead
requires exact clustering headers and ordered,
consecutive PCA/UMAP component columns after its barcode field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from spatialdata_io.readers._utils.errors import ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray
    from zarr.core.common import JSON

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers.xenium._version import Family

STRING = pa.string()
STRING_DTYPE = pd.ArrowDtype(STRING)
FLOAT64 = pa.float64()
FLOAT32 = pa.float32()
UINT64 = pa.uint64()
ALIGNMENT_TYPES = pa.schema([(str(index), FLOAT64) for index in range(3)])
SELECTION_TYPES = {"Selection": STRING, "X": FLOAT64, "Y": FLOAT64}

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
# Documented cells-table columns the reader relies on by name; XOA 2.0+ adds the
# segmentation method.
CELL_COLUMNS = (*CELL_REQUIRED, *CELL_OPTIONAL_COUNTS, "segmentation_method")
# Observation columns the reader adds; a cells table must not already have them.
READER_OBS = ("label_id", "nucleus_centroid_x", "nucleus_centroid_y", "region")

VERTICES = ("vertex_x", "vertex_y")
LABEL_ID = "label_id"

BARCODE = "Barcode"
CLUSTER = "Cluster"
_ANALYSIS_PROJECTION_PREFIXES = {"pca": "PC", "umap": "UMAP"}
_DECIMAL_ID = r"-1|[0-9]+"
_MIN_PROJECTION_COLUMNS = 2

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
    "codeword_index": pa.uint32(),
    "codeword_category": STRING,
    "is_gene": pa.bool_(),
}


def validate_header(names: Sequence[str], required: Sequence[str], *, path: Path, context: ReaderErrorContext) -> None:
    """Reject empty, duplicate or missing required column names."""
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ReaderFormatError(
            context=context, path=path, found=f"columns {list(names)!r}", expected="unique nonempty column names"
        )
    missing = [name for name in required if name not in names]
    if missing:
        raise ReaderFormatError(
            context=context, path=path, found=f"missing columns {missing!r}", expected=f"columns {list(required)!r}"
        )


def summary_columns(  # noqa: PLR0913, RUF100 - Native metadata and the external error boundary.
    shape: tuple[int, ...],
    dtype: np.dtype[np.generic],
    columns: JSON,
    n_cells: int,
    *,
    path: Path,
    context: ReaderErrorContext,
) -> tuple[str, ...]:
    """Require named float64 native summary storage before reading or converting values."""
    if not isinstance(columns, list) or shape != (n_cells, len(columns)):
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"cell_summary of shape {shape} with columns {columns!r}",
            expected="one named column per value",
        )
    if any(not isinstance(name, str) for name in columns):
        raise ReaderFormatError(
            context=context, path=path, found=f"cell_summary columns {columns!r}", expected="string column names"
        )
    names = tuple(name for name in columns if isinstance(name, str))
    validate_header(names, ("nucleus_centroid_x", "nucleus_centroid_y"), path=path, context=context)
    if dtype.kind != "f" or dtype.itemsize != np.dtype(np.float64).itemsize:
        raise ReaderFormatError(
            context=context, path=path, found=f"cell_summary dtype {dtype}", expected="float64 cell_summary storage"
        )
    return names


def validate_analysis_header(names: Sequence[str], kind: str, *, path: Path, context: ReaderErrorContext) -> None:
    """Require exact clustering fields or consecutive PCA/UMAP component fields."""
    names = list(names)
    if kind == "clustering":
        if names != [BARCODE, CLUSTER]:
            raise ReaderFormatError(context=context, path=path, found=f"header {names!r}", expected="Barcode,Cluster")
        return
    prefix = _ANALYSIS_PROJECTION_PREFIXES[kind]
    expected = [BARCODE, *(f"{prefix}-{index}" for index in range(1, len(names)))]
    if len(names) < _MIN_PROJECTION_COLUMNS or names != expected:
        raise ReaderFormatError(
            context=context, path=path, found=f"header {names!r}", expected=f"Barcode,{prefix}-1,..."
        )


def _id_type(family: Family) -> pa.DataType:
    """Return the identifier type: integers before XOA 1.3, text afterwards."""
    return UINT64 if family == "integer_ids" else STRING


def cell_types(family: Family) -> pa.Schema:
    """Return the declared types of the documented cells-table columns."""
    counts = (*CELL_REQUIRED_COUNTS, *CELL_OPTIONAL_COUNTS)
    return pa.schema(
        [(CELL_ID, _id_type(family)), *((name, FLOAT64) for name in CELL_FLOATS), *((name, UINT64) for name in counts)]
    )


def boundary_types(family: Family) -> pa.Schema:
    """Return the declared types of the boundary vertex columns."""
    return pa.schema(
        [(CELL_ID, _id_type(family)), (VERTICES[0], FLOAT32), (VERTICES[1], FLOAT32), (LABEL_ID, pa.uint32())]
    )


def transcript_types(family: Family) -> pa.Schema:
    """Return the declared transcript types, retaining signed unassigned ``-1`` IDs."""
    assignment_type = pa.int64() if family == "integer_ids" else STRING
    return pa.schema((TRANSCRIPT_REQUIRED | {CELL_ID: assignment_type} | TRANSCRIPT_OPTIONAL).items())


def header_types(names: Sequence[str], declared: pa.Schema) -> dict[str, pa.DataType]:
    """Return a type for every header column: the declared one, otherwise text."""
    return {name: declared.field(name).type if name in declared.names else STRING for name in names}


def transcript_columns(
    names: Sequence[str], declared: pa.Schema, *, path: Path, context: ReaderErrorContext
) -> dict[str, pa.DataType]:
    """Return the projected documented transcript columns of a header and their types."""
    validate_header(names, tuple(TRANSCRIPT_REQUIRED), path=path, context=context)
    return {field.name: field.type for field in declared if field.name in TRANSCRIPT_REQUIRED or field.name in names}


def cast_column(
    column: pa.ChunkedArray, kind: pa.DataType, name: str, *, path: Path, context: ReaderErrorContext
) -> pa.ChunkedArray:
    """Cast to the vendor type with Arrow's checked casts.

    Floating coordinates intentionally normalize to the declared float32 type;
    Arrow permits that rounding. Integer casts reject overflow and truncation.
    """
    if column.type == kind:
        return column
    try:
        return pc.cast(column, kind)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
        raise ReaderFormatError(
            context=context, path=path, found=f"{name} of type {column.type}", expected=f"{name} as {kind}"
        ) from error


def normalize_text(column: pa.ChunkedArray, name: str, *, path: Path, context: ReaderErrorContext) -> pa.ChunkedArray:
    """Return non-null ``string`` values; binary text is decoded strictly as UTF-8."""
    kind = column.type
    if not (
        pa.types.is_string(kind)
        or pa.types.is_large_string(kind)
        or pa.types.is_binary(kind)
        or pa.types.is_large_binary(kind)
    ):
        raise ReaderFormatError(
            context=context, path=path, found=f"{name} of type {kind}", expected=f"{name} as UTF-8 text"
        )
    try:
        # Arrow validates UTF-8 when casting binary to text.
        values = column if kind == STRING else pc.cast(column, STRING)
    except pa.ArrowInvalid as error:
        raise ReaderFormatError(
            context=context, path=path, found=f"{name} that is not UTF-8", expected=f"{name} as UTF-8 text"
        ) from error
    if values.null_count:
        raise ReaderFormatError(
            context=context, path=path, found=f"missing {name} values", expected=f"complete {name} values"
        )
    return values


def normalize_ids(
    column: pa.ChunkedArray, kind: pa.DataType, *, path: Path, context: ReaderErrorContext
) -> pa.ChunkedArray:
    """Return non-null identifiers of the declared type: positive ``uint64``, signed assignments or UTF-8 text."""
    if pa.types.is_string(kind):
        return normalize_text(column, CELL_ID, path=path, context=context)
    if not pa.types.is_integer(column.type) or column.null_count:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"identifiers of type {column.type}",
            expected="non-null integer identifiers",
        )
    values = cast_column(column, kind, CELL_ID, path=path, context=context)
    if pa.types.is_unsigned_integer(kind) and (values.to_numpy() == 0).any():
        raise ReaderFormatError(context=context, path=path, found="zero cell IDs", expected="positive cell IDs")
    return values


def decimal_ids(
    column: pa.ChunkedArray, kind: pa.DataType, *, path: Path, context: ReaderErrorContext
) -> pa.ChunkedArray:
    """Cast text cell IDs of a CSV to the signed ``kind``; only decimal digits or ``-1`` are accepted.

    Arrow's typed CSV grammar reads hexadecimal and wraps it in a signed type,
    so ``0xffffffffffffffff`` would become ``-1``, the unassigned sentinel.
    Signed CSV identifiers are therefore read as text and checked here; the
    cast rejects values outside ``kind``.
    """
    if not arrow_series(column).str.fullmatch(_DECIMAL_ID).all():
        raise ReaderFormatError(
            context=context,
            path=path,
            found="cell_id values other than decimal digits or -1",
            expected="decimal cell IDs or -1",
        )
    try:
        return pc.cast(column, kind)
    except pa.ArrowInvalid as error:
        raise ReaderFormatError(
            context=context, path=path, found=f"cell_id values outside {kind}", expected=f"cell_id as {kind}"
        ) from error


def arrow_series(column: pa.Array | pa.ChunkedArray) -> pd.Series:
    """View Arrow values as an Arrow-backed pandas series without copying them."""
    return pd.Series(pd.arrays.ArrowExtensionArray(column))


def id_index(column: pa.Array | pa.ChunkedArray) -> pd.Index:
    """Keep normalized identifiers in an Arrow-backed index for exact lookups."""
    return pd.Index(arrow_series(column))


def integer_ids(text: pd.Index, *, path: Path, context: ReaderErrorContext) -> pd.Index:
    """Parse canonical decimal identifiers (XOA 1.0-1.2 barcodes) to positive uint64 exactly."""
    try:
        values = id_index(pc.cast(pa.array(text), UINT64))
    except pa.ArrowInvalid as error:
        raise ReaderFormatError(
            context=context, path=path, found="non-integer cell IDs", expected="decimal integer cell IDs"
        ) from error
    if not values.astype("str").equals(text.astype("str")):
        raise ReaderFormatError(
            context=context, path=path, found="non-canonical integer cell IDs", expected="canonical decimal cell IDs"
        )
    if (values == 0).any():
        raise ReaderFormatError(context=context, path=path, found="zero cell IDs", expected="positive cell IDs")
    return values


def vocabulary_codes(
    values: pd.Series, vocabulary: pd.Index, name: str, *, path: Path, context: ReaderErrorContext
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
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"unknown {name} values {examples!r}",
            expected=f"{name} values from the count matrix",
        )
    return positions[codes]


def require_finite(
    values: NDArray[np.floating], name: str, *, path: Path, context: ReaderErrorContext, allow_nan: bool = False
) -> None:
    """Reject infinite values, and NaN unless the column documents it."""
    invalid = np.isinf(values) if allow_nan else ~np.isfinite(values)
    if invalid.any():
        raise ReaderFormatError(
            context=context, path=path, found=f"nonfinite {name} values", expected=f"finite {name} values"
        )


def require_no_nulls(table: pa.Table, *, path: Path, context: ReaderErrorContext) -> None:
    """Reject nulls; Xenium writes none in the projected columns."""
    missing = [name for name in table.column_names if table.column(name).null_count]
    if missing:
        raise ReaderFormatError(
            context=context, path=path, found=f"missing values in {missing!r}", expected="complete source records"
        )
