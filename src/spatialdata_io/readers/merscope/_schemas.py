"""MERSCOPE source columns, exact identities/counts, and physical annotations."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile

STRING_DTYPE = pd.ArrowDtype(pa.string())
FLOAT_DTYPE = pd.ArrowDtype(pa.float64())
INTEGER_DTYPE = pd.ArrowDtype(pa.int64())
_MAX_NUMBER_CHARACTERS = 128
_LITERAL_ZERO = r"[+-]?(?:0+(?:\.0*)?|\.0+)(?:[eE][+-]?\d+)?"
_BLANK_NAME = re.compile(r"Blank-[1-9][0-9]*")
_METADATA_FLOATS = frozenset(
    {
        "center_x",
        "center_y",
        "volume",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
        "anisotropy",
        "perimeter_area_ratio",
        "solidity",
    }
)
_TRANSCRIPT_OPTIONAL = ("global_z", "cell_id", "fov", "barcode_id", "transcript_id")
_OUTPUT_ANNOTATIONS = ("region", "slide", "dataset_id", "cells_region")


def schema_error(path: ArtifactFile, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a source contract violation with its reader and artifact identity."""
    return ReaderFormatError(
        context=ReaderErrorContext("merscope", path.name), path=path, found=found, expected=expected
    )


def validate_columns(columns: Sequence[str], path: ArtifactFile, *, first_empty: bool = False) -> None:
    """Reject duplicate names before parser repair, allowing an evidenced positional index."""
    if (
        not columns
        or len(set(columns)) != len(columns)
        or any("\x00" in name or (not name and not (first_empty and i == 0)) for i, name in enumerate(columns))
    ):
        raise schema_error(path, found="empty, duplicate or NUL-containing names", expected="unique named fields")


def count_columns(columns: Sequence[str], path: ArtifactFile) -> tuple[str, ...]:
    """Return ordered count features following the first positional cell identity field."""
    validate_columns(columns, path, first_empty=True)
    if not columns[1:]:
        raise schema_error(path, found="no count features", expected="a cell index followed by count features")
    return tuple(columns[1:])


def blank_columns(features: Sequence[str]) -> tuple[bool, ...]:
    """Identify only canonical positive-numbered Blank-N control names."""
    return tuple(_BLANK_NAME.fullmatch(feature) is not None for feature in features)


def normalize_cell_ids(series: pd.Series, path: ArtifactFile, *, allow_unassigned: bool = False) -> pd.Series:
    """Preserve exact nonempty lexical cell identities and the optional -1 sentinel."""
    if series.dtype.kind in "fb":
        raise schema_error(path, found="floating or boolean cell IDs", expected="exact text or integer cell identities")
    identifiers = series.astype(STRING_DTYPE)
    if identifiers.isna().any() or identifiers.eq("").any() or identifiers.str.contains("\x00", regex=False).any():
        raise schema_error(path, found="missing or malformed cell IDs", expected="nonempty exact cell identities")
    if not allow_unassigned and identifiers.eq("-1").any():
        raise schema_error(
            path, found="unassigned cell ID in a cell population", expected="cell identities excluding -1"
        )
    return identifiers


def integer_values(series: pd.Series, path: ArtifactFile) -> npt.NDArray[np.int64 | np.uint64]:
    """Decode nonnegative integral values without any floating intermediate."""
    lexical = series.astype(STRING_DTYPE)
    column = pa.array(lexical)
    if column.null_count:
        raise schema_error(path, found="missing or oversized integer", expected="nonnegative 64-bit integral values")
    try:
        # Arrow's native integer parser is the common count and stack-index path.
        values = pc.cast(column, pa.uint64()).to_numpy()
    except pa.ArrowInvalid:
        if lexical.str.len().gt(_MAX_NUMBER_CHARACTERS).any():
            raise schema_error(path, found="oversized integer", expected="nonnegative 64-bit integral values") from None
        exact_values: list[int] = []
        for token in lexical:
            try:
                exact = Decimal(token)
            except InvalidOperation as error:
                raise schema_error(path, found=f"integer {token!r}", expected="nonnegative integral values") from error
            if not exact.is_finite() or exact < 0 or exact >= 2**64 or exact != exact.to_integral_value():
                raise schema_error(
                    path, found=f"integer {token!r}", expected="nonnegative 64-bit integral values"
                ) from None
            exact_values.append(int(exact))
        dtype = np.uint64 if exact_values and max(exact_values) >= 2**63 else np.int64
        return np.asarray(exact_values, dtype=dtype)
    return values if len(values) and values.max() >= 2**63 else values.astype(np.int64, copy=False)


def normalize_floats(series: pd.Series, path: ArtifactFile, column: str) -> pd.Series:
    """Decode float64 physical values, rejecting nonfinite values and destructive conversion."""
    lexical = series.astype(STRING_DTYPE)
    expected = "finite float64 values without underflow or integer precision loss"
    if lexical.str.len().gt(_MAX_NUMBER_CHARACTERS).any():
        raise schema_error(path, found=f"oversized {column} values", expected=expected)
    try:
        normalized = lexical.astype(FLOAT_DTYPE)
    except (ValueError, TypeError, pa.ArrowInvalid) as error:
        raise schema_error(path, found=f"nonnumeric {column} values", expected=expected) from error
    values = normalized.to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(values).all():
        raise schema_error(path, found=f"nonfinite or missing {column} values", expected=expected)
    candidates = np.abs(values) >= 2**53
    zero = values == 0
    if zero.any():
        candidates |= zero & ~lexical.str.fullmatch(_LITERAL_ZERO).to_numpy(dtype=np.bool_)
    for row in np.flatnonzero(candidates):
        exact = Decimal(lexical.iloc[row])
        number = float(values[row])
        if (number == 0 and exact != 0) or (exact == exact.to_integral_value() and exact != Decimal(number)):
            raise schema_error(path, found=f"{column} precision loss", expected=expected)
    return normalized


def metadata_columns(columns: Sequence[str], path: ArtifactFile) -> tuple[str, ...]:
    """Require named metadata identities/centroids and preserve additional annotations."""
    validate_columns(columns, path)
    if not {"EntityID", "center_x", "center_y"}.issubset(columns):
        raise schema_error(
            path, found=f"columns {list(columns)!r}", expected="EntityID, center_x and center_y metadata"
        )
    return tuple(columns)


def normalize_metadata(frame: pd.DataFrame, path: ArtifactFile) -> pd.DataFrame:
    """Validate owned annotations, retaining unknown metadata as exact Arrow text."""
    identifiers = normalize_cell_ids(frame.pop("EntityID"), path)
    if not identifiers.is_unique:
        raise schema_error(path, found="duplicate metadata identities", expected="one metadata row per cell")
    frame.index = pd.Index(identifiers.to_numpy(), name=None)
    for column in frame.columns:
        if column in _METADATA_FLOATS:
            frame[column] = normalize_floats(frame[column], path, column)
    if "volume" in frame and frame["volume"].lt(0).any():
        raise schema_error(path, found="negative volume", expected="nonnegative cell volumes")
    if "transcript_count" in frame:
        values = integer_values(frame["transcript_count"], path)
        frame["transcript_count"] = pd.Series(
            values, index=frame.index, dtype=pd.ArrowDtype(pa.from_numpy_dtype(values.dtype))
        )
    if "fov" in frame:
        # Vendor metadata permits absent FOV assignments for merged entities.
        missing = frame["fov"].eq("")
        values = integer_values(frame.loc[~missing, "fov"], path)
        if values.dtype.kind == "u":
            raise schema_error(path, found="FOV overflow", expected="nullable nonnegative int64 FOV indexes")
        frame["fov"] = pd.Series(pd.NA, index=frame.index, dtype=INTEGER_DTYPE)
        frame.loc[~missing, "fov"] = values
    return frame


def validate_output_annotations(
    frame: pd.DataFrame, expected: Mapping[str, str], *, context: ReaderErrorContext
) -> None:
    """Reject conflicting source annotations before reader-owned names are attached."""
    for column in _OUTPUT_ANNOTATIONS:
        if column not in frame:
            continue
        value = expected.get(column)
        if value is None:
            raise ReaderFormatError(
                context=context,
                found=f"source reserved annotation {column!r} without a returned target",
                expected="no cells_region annotation on an unlinked table",
            )
        values = frame[column]
        if values.isna().any() or not values.eq(value).all():
            raise ReaderFormatError(
                context=context,
                found=f"conflicting source {column!r} values",
                expected=f"only the reader-owned value {value!r} for {column!r}",
            )


def boundary_columns(names: Sequence[str], path: ArtifactFile) -> tuple[str, ...]:
    """Project the source identity, plane, geometry and optional entity-type fields."""
    validate_columns(names, path)
    columns = ("EntityID", "ZIndex", "Geometry")
    if not set(columns).issubset(names):
        raise schema_error(path, found=f"columns {list(names)!r}", expected="EntityID, ZIndex and Geometry fields")
    return (*columns, "Type") if "Type" in names else columns


def normalize_boundary_ids(table: pa.Table, path: ArtifactFile) -> pd.Index:
    """Validate source primitives while retaining repeated rows for footprint union."""
    ids = table.column("EntityID")
    if not (pa.types.is_integer(ids.type) or pa.types.is_string(ids.type) or pa.types.is_large_string(ids.type)):
        raise schema_error(path, found=f"EntityID dtype {ids.type}", expected="exact integer or text identities")
    identifiers = normalize_cell_ids(pd.Series(ids, dtype=STRING_DTYPE), path)
    planes = integer_values(pd.Series(table.column("ZIndex"), dtype=STRING_DTYPE), path)
    if planes.dtype.kind == "u":
        raise schema_error(path, found="ZIndex overflow", expected="nonnegative int64 plane indexes")
    if "Type" in table.column_names:
        types = pd.Series(table.column("Type"), dtype=STRING_DTYPE)
        if types.isna().any() or not types.isin(("Cell", "cell")).all():
            raise schema_error(path, found="unsupported entity types", expected="only Cell/cell boundaries")
    if identifiers.empty:
        raise schema_error(path, found="no cell boundaries", expected="a nonempty cell footprint population")
    return pd.Index(identifiers.to_numpy(), name="EntityID")


def validate_transcript_columns(columns: Sequence[str], path: ArtifactFile) -> tuple[str, ...]:
    """Declare projected global coordinates, features and supported source annotations."""
    validate_columns(columns, path, first_empty=True)
    if not {"gene", "global_x", "global_y"}.issubset(columns):
        raise schema_error(
            path, found=f"columns {list(columns)!r}", expected="gene, global_x and global_y transcript fields"
        )
    return ("gene", "global_x", "global_y", *(column for column in _TRANSCRIPT_OPTIONAL if column in columns))


def normalize_transcript_partition(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    path: ArtifactFile,
    assignment: Literal["current", "original", "none"],
    allowed_cell_ids: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Validate one owned partition, preserving assignment provenance without a full scan."""
    genes = frame["gene"]
    if genes.isna().any() or not genes.isin(features).all():
        raise schema_error(
            path, found="missing or unknown gene names", expected="the complete source feature vocabulary"
        )
    frame["gene"] = pd.Categorical(genes.to_numpy(), categories=features)
    for column, axis in (("global_x", "x"), ("global_y", "y")):
        frame[axis] = normalize_floats(frame.pop(column), path, column).to_numpy(dtype=np.float64)
    for column in ("global_z", "fov", "barcode_id"):
        if column in frame:
            values = integer_values(frame.pop(column), path)
            if values.dtype.kind == "u":
                raise schema_error(path, found=f"{column} overflow", expected="nonnegative int64 source indexes")
            frame["z_raw" if column == "global_z" else column] = pd.Series(
                values, index=frame.index, dtype=INTEGER_DTYPE
            )
    if "transcript_id" in frame:
        values = frame["transcript_id"].astype(STRING_DTYPE)
        if values.isna().any() or values.eq("").any():
            raise schema_error(path, found="missing transcript_id", expected="nonempty codebook transcript identifiers")
        frame["transcript_id"] = values
    if "cell_id" in frame:
        identifiers = normalize_cell_ids(frame.pop("cell_id"), path, allow_unassigned=True)
        if (
            assignment == "current"
            and allowed_cell_ids is not None
            and not (identifiers.eq("-1") | identifiers.isin(allowed_cell_ids)).all()
        ):
            raise schema_error(
                path, found="unknown assigned cells", expected="current cell IDs or the -1 unassigned sentinel"
            )
        frame["original_cell_id" if assignment == "original" else "cell_id"] = identifiers
    return frame
