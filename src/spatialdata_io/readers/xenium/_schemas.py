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

from typing import TYPE_CHECKING, Literal

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
KEYPOINT_TYPES = pa.schema([(name, FLOAT64) for name in ("fixedX", "fixedY", "alignmentX", "alignmentY")])
SELECTION_TYPES = {"Selection": STRING, "X": FLOAT64, "Y": FLOAT64, "Class": STRING, "Color": STRING}

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
BOUNDARY_REQUIRED = (CELL_ID, *VERTICES)

BARCODE = "Barcode"
CLUSTER = "Cluster"
type AnalysisKind = Literal["clustering", "pca", "umap"]
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


def analysis_schema(names: Sequence[str], kind: AnalysisKind, *, path: Path, context: ReaderErrorContext) -> pa.Schema:
    """Return the typed projection for exact clustering or consecutive PCA/UMAP fields."""
    names = list(names)
    if kind == "clustering":
        if names != [BARCODE, CLUSTER]:
            raise ReaderFormatError(context=context, path=path, found=f"header {names!r}", expected="Barcode,Cluster")
        return pa.schema([(BARCODE, STRING), (CLUSTER, UINT64)])
    prefix = _ANALYSIS_PROJECTION_PREFIXES[kind]
    expected = [BARCODE, *(f"{prefix}-{index}" for index in range(1, len(names)))]
    if len(names) < _MIN_PROJECTION_COLUMNS or names != expected:
        raise ReaderFormatError(
            context=context, path=path, found=f"header {names!r}", expected=f"Barcode,{prefix}-1,..."
        )
    return pa.schema([(BARCODE, STRING), *((name, FLOAT64) for name in expected[1:])])


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


def boundary_columns(names: Sequence[str], *, path: Path, context: ReaderErrorContext) -> tuple[str, ...]:
    """Select documented vertex fields and the mask label when present."""
    validate_header(names, BOUNDARY_REQUIRED, path=path, context=context)
    return (*BOUNDARY_REQUIRED, *((LABEL_ID,) if LABEL_ID in names else ()))


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


def validate_parquet_types(source: pa.Schema, declared: pa.Schema, *, path: Path, context: ReaderErrorContext) -> None:
    """Check native physical types of projected fields without decoding values."""
    for field in declared:
        index = source.get_field_index(field.name)
        if index == -1:
            continue
        stored = source.field(index).type
        if not _compatible_parquet_type(stored, field):
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"{field.name} of type {stored} in Parquet footer",
                expected=f"a source representation of {field.name} as {field.type}",
            )


def _text_storage(kind: pa.DataType) -> bool:
    return (
        pa.types.is_string(kind)
        or pa.types.is_large_string(kind)
        or pa.types.is_binary(kind)
        or pa.types.is_large_binary(kind)
    )


def _compatible_parquet_type(stored: pa.DataType, field: pa.Field) -> bool:
    """Use the case-specific representation rules used by the value decoders."""
    if stored == field.type:
        return True
    if pa.types.is_string(field.type):
        return _text_storage(stored)
    if field.name == CELL_ID:
        return pa.types.is_integer(stored)
    if pa.types.is_boolean(field.type):
        return False
    # Numeric fields use Arrow's checked casts. A dictionary of native
    # numbers remains numeric, and null-only optional columns are valid.
    physical = stored.value_type if isinstance(stored, pa.DictionaryType) else stored
    return pa.types.is_integer(physical) or pa.types.is_floating(physical) or pa.types.is_null(physical)


def normalize_text(column: pa.ChunkedArray, name: str, *, path: Path, context: ReaderErrorContext) -> pa.ChunkedArray:
    """Return non-null ``string`` values; binary text is decoded strictly as UTF-8."""
    kind = column.type
    if not _text_storage(kind):
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


def validate_membership(
    values: pd.Series, vocabulary: pd.Index, name: str, *, path: Path, context: ReaderErrorContext
) -> None:
    """Reject values absent from the vocabulary, looking up only distinct source values."""
    unique = pd.Index(values.unique())
    unknown = vocabulary.get_indexer(unique) < 0
    if unknown.any():
        examples = unique[unknown][:5].tolist()
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"unknown {name} values {examples!r}",
            expected=f"{name} values from the count matrix",
        )


def require_finite(
    values: NDArray[np.floating], name: str, *, path: Path, context: ReaderErrorContext, allow_nan: bool = False
) -> None:
    """Reject infinite values, and NaN unless the column documents it."""
    invalid = np.isinf(values) if allow_nan else ~np.isfinite(values)
    if invalid.any():
        raise ReaderFormatError(
            context=context, path=path, found=f"nonfinite {name} values", expected=f"finite {name} values"
        )


def normalize_cell_table(
    table: pa.Table, declared: pa.Schema, *, path: Path, context: ReaderErrorContext
) -> tuple[pa.Table, pd.Index]:
    """Check cell annotation domains and preserve the complete Arrow table order."""
    for field in declared:
        index = table.schema.get_field_index(field.name)
        if index == -1 or field.name == CELL_ID:
            continue
        column = cast_column(table[field.name], field.type, field.name, path=path, context=context)
        if field.name == "nucleus_area":
            # Writers store missing nucleus areas as null or NaN. Normalize both to NaN.
            column = column.fill_null(float("nan"))
        table = table.set_column(index, field.name, column)

    cell_column = normalize_ids(table[CELL_ID], declared.field(CELL_ID).type, path=path, context=context)
    table = table.set_column(table.schema.get_field_index(CELL_ID), CELL_ID, cell_column)
    if any(table[name].null_count for name in CELL_REQUIRED):
        raise ReaderFormatError(
            context=context, path=path, found="missing required values", expected="complete cell records"
        )
    for name in CELL_FLOATS:
        values = table[name].to_numpy()
        require_finite(values, name, path=path, context=context, allow_nan=name == "nucleus_area")
        if name in ("cell_area", "nucleus_area") and (values < 0).any():
            raise ReaderFormatError(
                context=context, path=path, found=f"negative {name} values", expected=f"nonnegative {name}"
            )

    ids = id_index(cell_column)
    if not ids.is_unique:
        raise ReaderFormatError(context=context, path=path, found="repeated cell IDs", expected="one row per cell")
    return table, ids


def require_no_nulls(table: pa.Table, *, path: Path, context: ReaderErrorContext) -> None:
    """Reject nulls; Xenium writes none in the projected columns."""
    missing = [name for name in table.column_names if table.column(name).null_count]
    if missing:
        raise ReaderFormatError(
            context=context, path=path, found=f"missing values in {missing!r}", expected="complete source records"
        )
