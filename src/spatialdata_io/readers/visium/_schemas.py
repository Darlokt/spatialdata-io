"""Typed Visium spot positions and their barcode, tissue and coordinate invariants.

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
Arrow reads barcodes literally, tissue state as uint8, grid positions as
uint64 and image-pixel positions directly as float64. Tissue state is binary;
pixel positions must be finite and nonnegative. Numeric token grammar and
rounding follow Arrow's declared types, without a second text decoder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

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
POSITION_SOURCE_SCHEMA = pa.schema(
    [
        ("barcode", pa.string()),
        ("in_tissue", pa.uint8()),
        ("array_row", pa.uint64()),
        ("array_col", pa.uint64()),
        ("pxl_row_in_fullres", pa.float64()),
        ("pxl_col_in_fullres", pa.float64()),
    ]
)
REGION_KEY = "region"
INSTANCE_KEY = "spot_id"
SPOT_ID_DTYPE = pd.ArrowDtype(pa.string())


def classify_position_row(columns: Sequence[str], *, path: Path, context: ReaderErrorContext) -> PositionFormat:
    """Recognize named fields before CSV parsers can repair duplicate headers.

    Named fields may be reordered and extended; additional fields are projected
    away. Headerless input retains exactly six positional fields.
    """
    if any(column in POSITION_COLUMNS for column in columns):
        if any(not column for column in columns) or len(set(columns)) != len(columns):
            raise ReaderFormatError(
                context=context,
                path=path,
                found="empty or duplicate column names",
                expected="nonempty unique column names",
            )
        if not set(POSITION_COLUMNS).issubset(columns):
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"columns {tuple(columns)!r}",
                expected=f"required columns {POSITION_COLUMNS!r}",
            )
        return "headered"

    if len(columns) != len(POSITION_COLUMNS):
        raise ReaderFormatError(
            context=context, path=path, found=f"{len(columns)} fields", expected="exactly six positional fields"
        )

    return "headerless"


def normalize_positions(frame: pd.DataFrame, *, path: Path, context: ReaderErrorContext) -> pd.DataFrame:
    """Normalize owned projected source fields; preserve barcode and row order.

    Source fields have their declared Arrow types. Barcodes must be nonempty
    and unique, pixel coordinates finite and nonnegative, and in_tissue binary.
    No source rows are filtered.
    """
    if frame.empty:
        raise ReaderFormatError(context=context, path=path, found="zero rows", expected="at least one spot row")

    barcodes = frame["barcode"]
    if barcodes.isna().any() or (barcodes.str.len() == 0).any() or barcodes.str.contains("\x00", regex=False).any():
        raise ReaderFormatError(
            context=context,
            path=path,
            found="null, empty, or NUL-containing barcodes",
            expected="nonempty barcode strings without NUL",
        )
    if not barcodes.is_unique:
        duplicates = tuple(barcodes[barcodes.duplicated()].drop_duplicates().head(5).tolist())
        raise ReaderFormatError(
            context=context, path=path, found=f"duplicate barcodes {duplicates!r}", expected="unique barcode strings"
        )

    if not frame["in_tissue"].isin([0, 1]).all():
        raise ReaderFormatError(
            context=context, path=path, found="in_tissue values outside {0, 1}", expected="binary in_tissue values"
        )

    for column in POSITION_COLUMNS[-2:]:
        values = pa.array(frame[column])
        if values.null_count or pc.call_function("all", [pc.call_function("is_finite", [values])]).as_py() is False:
            raise ReaderFormatError(
                context=context, path=path, found=f"nonfinite values in {column!r}", expected="finite numeric values"
            )
        if (frame[column] < 0).any():
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"negative values in {column!r}",
                expected="nonnegative pixel coordinates",
            )

    return frame.set_index("barcode")
