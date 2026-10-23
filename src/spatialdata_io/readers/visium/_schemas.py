"""Visium positional and named spot schemas and exact numeric normalization."""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pyarrow as pa

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from pathlib import Path

type PositionFormat = Literal["headered", "headerless"]

POSITION_COLUMNS = (
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
)
OBS_POSITION_COLUMNS = ("in_tissue", "array_row", "array_col")
POSITION_SOURCE_SCHEMA = pa.schema([(column, pa.string()) for column in POSITION_COLUMNS])
REGION_KEY = "region"
INSTANCE_KEY = "spot_id"
SPOT_ID_DTYPE = pd.ArrowDtype(pa.string())


def position_error(path: Path, found: str, expected: str) -> ReaderFormatError:
    """Describe a malformed position artifact at its external boundary."""
    return ReaderFormatError(
        context=ReaderErrorContext("visium", "tissue positions"), path=path, found=found, expected=expected
    )


def classify_position_row(row: str, *, path: Path) -> PositionFormat:
    """Recognize named fields before CSV parsers can repair duplicate headers.

    Named fields may be reordered and extended; additional fields are projected
    away. Headerless input retains exactly six positional fields.
    """
    try:
        columns = next(csv.reader([row], strict=True))
    except csv.Error as error:
        raise position_error(path, "malformed first CSV row", "a valid header or six-field data row") from error

    if any(column in POSITION_COLUMNS for column in columns):
        if any(not column for column in columns) or len(set(columns)) != len(columns):
            raise position_error(path, "empty or duplicate column names", "nonempty unique column names")
        if not set(POSITION_COLUMNS).issubset(columns):
            raise position_error(path, f"columns {tuple(columns)!r}", f"required columns {POSITION_COLUMNS!r}")
        return "headered"

    if len(columns) != len(POSITION_COLUMNS):
        raise position_error(path, f"{len(columns)} fields", "exactly six positional fields")

    return "headerless"


def _grid_column(source: pd.Series, *, path: Path) -> pd.Series:
    """Parse ordinary grid integers natively and decimal variants exactly."""
    if source.isna().any():
        raise position_error(path, f"null values in {source.name!r}", "finite numeric values")

    try:
        numeric = source.astype("int64[pyarrow]")
    except pa.ArrowInvalid:
        # Space Ranger uses integer tokens. Accept decimal/exponent variants
        # through exact decimal arithmetic, never a rounded float intermediate.
        values = [_grid_value(text, column=str(source.name), path=path) for text in source]
        return pd.Series(values, index=source.index, dtype="int64[pyarrow]", name=source.name)

    if (numeric < 0).any():
        raise position_error(path, f"negative values in {source.name!r}", "nonnegative integer values")

    return numeric


def _grid_value(text: str, *, column: str, path: Path) -> int:
    """Validate an uncommon numeric token before constructing an exact int64."""
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise position_error(path, f"nonnumeric values in {column!r}", "finite numeric values") from error

    if not value.is_finite():
        raise position_error(path, f"nonfinite values in {column!r}", "finite numeric values")
    if value != value.to_integral_value():
        raise position_error(path, f"fractional values in {column!r}", "integer values")
    if value < 0:
        raise position_error(path, f"negative values in {column!r}", "nonnegative integer values")
    if value > np.iinfo(np.int64).max:
        raise position_error(path, f"overflow in {column!r}", "integer values fitting int64")

    return int(value)


def _pixel_column(source: pd.Series, *, path: Path) -> pd.Series:
    """Parse correctly rounded float64 coordinates and reject unsafe loss."""
    if source.isna().any():
        raise position_error(path, f"null values in {source.name!r}", "finite numeric values")

    try:
        numeric = source.astype("float64[pyarrow]")
    except pa.ArrowInvalid as error:
        raise position_error(path, f"nonnumeric values in {source.name!r}", "finite numeric values") from error

    values = numeric.to_numpy(copy=False)
    if not np.isfinite(values).all():
        raise position_error(path, f"nonfinite values in {source.name!r}", "finite numeric values")
    if (values < 0).any():
        raise position_error(path, f"negative values in {source.name!r}", "nonnegative pixel coordinates")

    # Beyond exact-integer precision, or at underflow, compare source text
    # directly so a valid coordinate cannot silently move to a different pixel.
    for row in np.flatnonzero((values >= 2**53) | (values == 0)):
        if Decimal(source.iloc[row]) != Decimal(float(values[row])):
            raise position_error(
                path, f"pixel precision loss in {source.name!r}", "losslessly representable coordinates"
            )

    return numeric


def normalize_positions(frame: pd.DataFrame, *, path: Path) -> pd.DataFrame:
    """Normalize owned projected source fields; preserve barcode and row order.

    Source fields are Arrow strings selected by the storage parser. Barcodes
    must be nonempty and unique, grid fields nonnegative int64, and pixel
    coordinates finite nonnegative float64. Only in_tissue is restricted to
    the binary values zero and one. No source rows are filtered.
    """
    if frame.empty:
        raise position_error(path, "zero rows", "at least one spot row")

    barcodes = frame["barcode"]
    if barcodes.isna().any() or (barcodes.str.len() == 0).any() or barcodes.str.contains("\x00", regex=False).any():
        raise position_error(path, "null, empty, or NUL-containing barcodes", "nonempty barcode strings without NUL")
    if not barcodes.is_unique:
        duplicates = tuple(barcodes[barcodes.duplicated()].drop_duplicates().head(5).tolist())
        raise position_error(path, f"duplicate barcodes {duplicates!r}", "unique barcode strings")

    for column in OBS_POSITION_COLUMNS:
        frame[column] = _grid_column(frame[column], path=path)

    if not frame["in_tissue"].isin([0, 1]).all():
        raise position_error(path, "in_tissue values outside {0, 1}", "binary in_tissue values")

    for column in POSITION_COLUMNS[-2:]:
        frame[column] = _pixel_column(frame[column], path=path)

    return frame.set_index("barcode")
