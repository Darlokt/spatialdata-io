"""CODEX source fields, dynamic cycle channels, exact IDs, and Arrow annotations.

Required cell fields are named and unordered; every extra non-cycle field is
preserved. Only complete cycNNN_chNNN names denote expression. CSV identifiers
remain lexical until exact signed/unsigned 64-bit normalization. FCS primitive
numeric widths are retained. Optional metadata may be missing; required IDs,
coordinates, regions, and compensated intensities must be valid.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile

STRING_DTYPE = pd.ArrowDtype(pa.string())
FLOAT_DTYPE = pd.ArrowDtype(pa.float64())
INTEGER_DTYPES = {"i": pd.ArrowDtype(pa.int64()), "u": pd.ArrowDtype(pa.uint64())}
REQUIRED_COLUMNS = frozenset({"cell_id", "region", "x", "y"})
_CYCLE_COLUMN = re.compile(r"cyc\d+_ch\d+")
_ZERO_PATTERN = r"[+-]?(?:0+(?:\.0*)?|\.0+)(?:[eE][+-]?\d+)?"


def schema_error(path: ArtifactFile, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a source-schema violation with its CODEX artifact identity."""
    return ReaderFormatError(context=ReaderErrorContext("codex", path.name), path=path, found=found, expected=expected)


def cell_columns(columns: list[str], path: ArtifactFile) -> tuple[list[str], list[str]]:
    """Validate source names and separate cycle features from metadata in source order."""
    if (
        not columns
        or any(not isinstance(column, str) or not column or "\x00" in column for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise schema_error(
            path, found="empty or duplicate column names", expected="nonempty unique source column names"
        )

    if not REQUIRED_COLUMNS.issubset(columns):
        raise schema_error(
            path, found=f"columns {columns!r}", expected=f"required columns {sorted(REQUIRED_COLUMNS)!r}"
        )

    features = [column for column in columns if _CYCLE_COLUMN.fullmatch(column)]
    if not features:
        raise schema_error(path, found="no cycle channels", expected="at least one cycNNN_chNNN expression column")

    feature_set = set(features)
    return features, [column for column in columns if column not in feature_set]


def csv_column_names(columns: list[str], path: ArtifactFile) -> tuple[list[str], list[str] | None]:
    """Normalize complete Akoya short-name:long-label headers in source order.

    Plain normalized tables retain additional metadata names literally. A
    qualified export must qualify every field; duplicate short names remain
    errors at the ordinary cell-column boundary. Split only the first colon.
    """
    if REQUIRED_COLUMNS.issubset(columns):
        return columns, None
    if not columns or not all(":" in column for column in columns):
        return columns, None
    names: list[str] = []
    labels: list[str] = []
    for column in columns:
        name, _, label = column.partition(":")
        if not label or "\x00" in label:
            raise schema_error(path, found="empty or NUL-containing long label", expected="nonempty Akoya long labels")
        names.append(name)
        labels.append(label)
    return names, labels


def check_integer_promotion(values: npt.NDArray[np.number], path: ArtifactFile) -> None:
    """Reject integers that float64 expression or geometry storage would change."""
    if values.dtype.kind in "iu":
        large = values > 2**53
        if values.dtype.kind == "i":
            large |= values < -(2**53)

        for value in values[large]:
            if int(value) != int(float(value)):
                raise schema_error(
                    path, found="integer-to-float precision loss", expected="exactly representable integer values"
                )


def cell_ids(values: pd.Series, path: ArtifactFile) -> npt.NDArray[np.int64 | np.uint64]:
    """Convert a nonempty population's integral IDs exactly without mutating it.

    Choose uint64 only when required by the largest identifier. Reject values
    that cannot share one signed or unsigned 64-bit representation.
    """
    # Only integer parser results can bypass lexical checks: decimal tokens can
    # promote another exact identifier to an inexact float64 value.
    if not values.isna().any():
        try:
            numeric = pd.to_numeric(values)
        except ValueError:
            numeric = None
        if numeric is not None and numeric.dtype.kind in "iu":
            largest = int(numeric.max())
            dtype = np.uint64 if largest >= 2**63 else np.int64
            return numeric.to_numpy(dtype=dtype, copy=False)

    # For FCS, Decimal(float) retains the stored binary value, unlike str(float).
    identifiers: list[int] = []
    for value in values:
        try:
            exact = Decimal(float(value)) if isinstance(value, (float, np.floating)) else Decimal(str(value))
        except InvalidOperation as error:
            raise schema_error(path, found=f"cell_id {value!r}", expected="finite integral cell identifiers") from error
        if not exact.is_finite() or exact != exact.to_integral_value() or exact < -(2**63) or exact >= 2**64:
            raise schema_error(path, found=f"cell_id {value!r}", expected="finite 64-bit integral cell identifiers")
        identifiers.append(int(exact))

    if min(identifiers) < 0 and max(identifiers) >= 2**63:
        raise schema_error(path, found="mixed signed/unsigned overflow", expected="one representable 64-bit ID dtype")
    return np.asarray(identifiers, dtype=np.uint64 if max(identifiers) >= 2**63 else np.int64)


def region_name(regions: pd.Series, path: ArtifactFile) -> str:
    """Require one region and normalize positive integral values to string names."""
    if regions.isna().any() or regions.ne(regions.iloc[0]).any():
        raise schema_error(path, found="missing or multiple regions", expected="one non-null region per cell table")

    region = regions.iloc[0]
    if isinstance(region, (int, float, np.integer, np.floating)):
        if isinstance(region, bool) or not np.isfinite(region) or region <= 0 or region != int(region):
            raise schema_error(path, found=f"region {region!r}", expected="a positive integer region or nonempty name")
        return str(int(region))

    if isinstance(region, str) and region:
        return region
    raise schema_error(path, found=f"region {region!r}", expected="a positive integer region or nonempty name")


def normalize_csv_numbers(frame: pd.DataFrame, columns: list[str], path: ArtifactFile) -> None:
    """Normalize lexical CSV columns into explicit Arrow primitive storage.

    Cell IDs remain lexical until population-wide integer normalization. Extra
    metadata remains text when it is not wholly numeric. Empty numeric fields
    become missing, while physically absent fields have already failed parsing.
    """
    for column in columns:
        source = frame[column]
        if source.str.contains("\x00", regex=False).any():
            raise schema_error(
                path, found=f"NUL character in {column!r}", expected="text fields without NUL characters"
            )
        if column == "cell_id":
            continue

        try:
            numeric = pd.to_numeric(source)
        except ValueError:
            continue
        if numeric.dtype.kind in INTEGER_DTYPES:
            frame[column] = numeric.astype(INTEGER_DTYPES[numeric.dtype.kind])
            continue

        try:
            numeric = source.replace("", pd.NA).astype(FLOAT_DTYPE)
        except (ValueError, pa.ArrowInvalid):
            continue

        values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
        # Sparse signal blocks contain many literal zeros. Only a nonzero
        # lexical value rounded to zero needs an underflow check.
        underflow = (values == 0) & ~source.str.fullmatch(_ZERO_PATTERN).to_numpy(dtype=np.bool_)
        candidates = source.ne("").to_numpy() & ((np.abs(values) >= 2**53) | underflow)
        for row in np.flatnonzero(candidates):
            exact = Decimal(source.iloc[row])
            if exact.is_finite() and exact != Decimal(float(values[row])):
                raise schema_error(
                    path, found="CSV numeric precision loss", expected="losslessly representable numeric values"
                )

        frame[column] = numeric


def normalize_fcs_metadata(frame: pd.DataFrame, path: ArtifactFile) -> None:
    """Keep owned FCS annotations and primitive widths in explicit Arrow columns."""
    for column, dtype in frame.dtypes.items():
        if dtype.kind not in "iuf":
            raise schema_error(path, found=f"FCS parameter dtype {dtype}", expected="real numeric event parameters")

        frame[column] = frame[column].astype(pd.ArrowDtype(pa.from_numpy_dtype(dtype)))


def expression_dtype(frame: pd.DataFrame, features: list[str], path: ArtifactFile) -> np.dtype:
    """Choose a real kernel dtype for lexical CSV results or native FCS buffers."""
    dtypes: list[np.dtype] = []
    for column in features:
        dtype = frame[column].dtype
        if isinstance(dtype, pd.ArrowDtype):
            dtype = np.dtype(dtype.pyarrow_dtype.to_pandas_dtype())

        if not isinstance(dtype, np.dtype) or dtype.kind not in "iuf":
            raise schema_error(
                path, found="nonnumeric cycle values or x/y coordinates", expected="finite real numeric values"
            )
        dtypes.append(dtype)

    return np.result_type(*dtypes)


def validate_observations(observations: pd.DataFrame, path: ArtifactFile) -> npt.NDArray[np.number]:
    """Validate normalized cell identity and exact finite region-pixel coordinates."""
    if not observations["cell_id"].is_unique:
        raise schema_error(path, found="duplicate cell_id values", expected="unique cell identifiers within one region")

    # Geometry stores float64 centers even when the input coordinates are ints.
    for axis in ("x", "y"):
        check_integer_promotion(observations[axis].to_numpy(copy=False), path)
    coordinates = observations[["x", "y"]].to_numpy(dtype=expression_dtype(observations, ["x", "y"], path))
    if not np.isfinite(coordinates).all():
        raise schema_error(path, found="invalid x/y coordinates", expected="finite real region pixel coordinates")

    return coordinates
