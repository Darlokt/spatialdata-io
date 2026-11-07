"""Schemas of the Visium spot positions and their exact numeric normalization.

Every capture spot has one row, in or out of the tissue (in Space Ranger 1.x
headerless, in 2.0 and later with this header)::

    barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres
    ACGCCTGACACGCGCT-1,0,0,0,1234.5,2345.6

    barcode               spot barcode: nonempty, unique, without NUL
    in_tissue             1 under tissue, 0 otherwise
    array_row, array_col  spot row and column on the capture array (>= 0)
    pxl_row_in_fullres    spot centre y in full-resolution image pixels
    pxl_col_in_fullres    spot centre x in full-resolution image pixels

A header names its fields, so they may come in any order and further columns
are ignored; empty or duplicate names are rejected before a CSV parser could
rename them. A headerless file has exactly these six fields in this order.
Fields are read as text and then normalized: the grid fields as exact int64
integer tokens (``2.0`` and ``2e0`` are rejected), the pixel fields as
correctly rounded finite nonnegative float64, which may be fractional. A
pixel value above 2**53 or one that rounds to zero must be exactly
representable, so that no spot silently moves to another pixel.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pyarrow as pa

from spatialdata_io.readers._utils.errors import ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from spatialdata_io.readers._utils.errors import ReaderErrorContext

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


def position_error(context: ReaderErrorContext, path: Path, found: str, expected: str) -> ReaderFormatError:
    """Describe a malformed position artifact at its external boundary."""
    return ReaderFormatError(context=context, path=path, found=found, expected=expected)


def classify_position_row(columns: Sequence[str], *, path: Path, context: ReaderErrorContext) -> PositionFormat:
    """Recognize named fields before CSV parsers can repair duplicate headers.

    Named fields may be reordered and extended; additional fields are projected
    away. Headerless input retains exactly six positional fields.
    """
    if any(column in POSITION_COLUMNS for column in columns):
        if any(not column for column in columns) or len(set(columns)) != len(columns):
            raise position_error(context, path, "empty or duplicate column names", "nonempty unique column names")
        if not set(POSITION_COLUMNS).issubset(columns):
            raise position_error(context, path, f"columns {tuple(columns)!r}", f"required columns {POSITION_COLUMNS!r}")
        return "headered"

    if len(columns) != len(POSITION_COLUMNS):
        raise position_error(context, path, f"{len(columns)} fields", "exactly six positional fields")

    return "headerless"


def _grid_column(source: pd.Series, *, path: Path, context: ReaderErrorContext) -> pd.Series:
    """Parse integer tokens exactly, without accepting decimal/exponent spellings."""
    if source.isna().any():
        raise position_error(context, path, f"null values in {source.name!r}", "finite numeric values")

    try:
        numeric = source.astype("int64[pyarrow]")
    except pa.ArrowInvalid as error:
        raise position_error(
            context, path, f"invalid integer token or overflow in {source.name!r}", "integer tokens fitting int64"
        ) from error

    if (numeric < 0).any():
        raise position_error(context, path, f"negative values in {source.name!r}", "nonnegative integer values")

    return numeric


def _pixel_column(source: pd.Series, *, path: Path, context: ReaderErrorContext) -> pd.Series:
    """Parse correctly rounded float64 coordinates and reject unsafe loss."""
    if source.isna().any():
        raise position_error(context, path, f"null values in {source.name!r}", "finite numeric values")

    try:
        numeric = source.astype("float64[pyarrow]")
    except pa.ArrowInvalid as error:
        raise position_error(context, path, f"nonnumeric values in {source.name!r}", "finite numeric values") from error

    values = numeric.to_numpy(copy=False)
    if not np.isfinite(values).all():
        raise position_error(context, path, f"nonfinite values in {source.name!r}", "finite numeric values")
    if (values < 0).any():
        raise position_error(context, path, f"negative values in {source.name!r}", "nonnegative pixel coordinates")

    # Beyond exact-integer precision, or at underflow, compare source text
    # directly so a valid coordinate cannot silently move to a different pixel.
    for row in np.flatnonzero((values >= 2**53) | (values == 0)):
        if Decimal(source.iloc[row]) != Decimal(float(values[row])):
            raise position_error(
                context, path, f"pixel precision loss in {source.name!r}", "losslessly representable coordinates"
            )

    return numeric


def normalize_positions(frame: pd.DataFrame, *, path: Path, context: ReaderErrorContext) -> pd.DataFrame:
    """Normalize owned projected source fields; preserve barcode and row order.

    Source fields are Arrow strings selected by the storage parser. Barcodes
    must be nonempty and unique, grid fields nonnegative int64, and pixel
    coordinates finite nonnegative float64. Only in_tissue is restricted to
    the binary values zero and one. No source rows are filtered.
    """
    if frame.empty:
        raise position_error(context, path, "zero rows", "at least one spot row")

    barcodes = frame["barcode"]
    if barcodes.isna().any() or (barcodes.str.len() == 0).any() or barcodes.str.contains("\x00", regex=False).any():
        raise position_error(
            context, path, "null, empty, or NUL-containing barcodes", "nonempty barcode strings without NUL"
        )
    if not barcodes.is_unique:
        duplicates = tuple(barcodes[barcodes.duplicated()].drop_duplicates().head(5).tolist())
        raise position_error(context, path, f"duplicate barcodes {duplicates!r}", "unique barcode strings")

    for column in OBS_POSITION_COLUMNS:
        frame[column] = _grid_column(frame[column], path=path, context=context)

    if not frame["in_tissue"].isin([0, 1]).all():
        raise position_error(context, path, "in_tissue values outside {0, 1}", "binary in_tissue values")

    for column in POSITION_COLUMNS[-2:]:
        frame[column] = _pixel_column(frame[column], path=path, context=context)

    return frame.set_index("barcode")
