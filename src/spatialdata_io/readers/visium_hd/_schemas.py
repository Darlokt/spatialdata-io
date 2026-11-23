"""Visium HD tabular contracts for positions, barcodes and feature identity.

Every bin size has one ``spatial/tissue_positions.parquet`` with one row per
grid square, including squares outside the tissue and without counts::

    column              Arrow type  meaning
    barcode             string      s_<size:03d>um_<row:05d>_<col:05d>-1
    in_tissue           uint8       0 or 1, the tissue mask
    array_row           uint32      bin row on the grid of that size
    array_col           uint32      bin column on the grid of that size
    pxl_row_in_fullres  float64     image y of the bin, full-resolution pixels
    pxl_col_in_fullres  float64     image x of the bin, full-resolution pixels

Space Ranger 4.0 and later also write ``barcode_mappings.parquet``, one row per
2 µm square, assigning it to a segmented cell::

    column        Arrow type  meaning
    square_002um  string      the 2 µm barcode
    cell_id       string      cellid_<id:09d>-1, null outside cells
    in_cell       bool        the square lies in a cell
    in_nucleus    bool        the square lies in that cell's nucleus

Further ``square_XXXum`` columns name the coarser bins of the same square and
are not read. Named fields may appear in any order beside extra columns;
duplicate column names are rejected. Values must be nonnull (``cell_id``
exactly where ``in_cell`` holds), pixel positions finite, and a nucleus square
must lie in a cell. See
https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/spatial-outputs
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

POSITIONS_SCHEMA = pa.schema(
    [
        pa.field("barcode", pa.string(), nullable=False),
        pa.field("in_tissue", pa.uint8(), nullable=False),
        pa.field("array_row", pa.uint32(), nullable=False),
        pa.field("array_col", pa.uint32(), nullable=False),
        pa.field("pxl_row_in_fullres", pa.float64(), nullable=False),
        pa.field("pxl_col_in_fullres", pa.float64(), nullable=False),
    ]
)
MAPPINGS_SCHEMA = pa.schema(
    [
        pa.field("square_002um", pa.string(), nullable=False),
        pa.field("cell_id", pa.string()),
        pa.field("in_cell", pa.bool_(), nullable=False),
        pa.field("in_nucleus", pa.bool_(), nullable=False),
    ]
)
NUCLEUS_FEATURE_COLUMNS = ("gene_ids", "genome", "feature_types")
MAX_CELL_ID = 999_999_999


def validate_nucleus_feature_columns(frame: pd.DataFrame, path: Path, *, context: ReaderErrorContext) -> None:
    """Require the ordered feature identity annotations used across nucleus count sources."""
    missing = tuple(name for name in NUCLEUS_FEATURE_COLUMNS if name not in frame)
    if missing:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"missing feature columns {missing!r}",
            expected=repr(NUCLEUS_FEATURE_COLUMNS),
        )


def bin_instance_ids(count: int) -> NDArray[np.uint32]:
    """Generate one uint32 row ID per selected bin, reserving zero for label background."""
    if count > np.iinfo(np.uint32).max:
        message = "The number of bin instances exceeds uint32 label capacity."
        raise ValueError(message)
    return np.arange(1, count + 1, dtype=np.uint32)


def parse_cell_barcodes(names: pd.Index[str], path: Path, *, version: str | None) -> NDArray[np.uint32]:
    """Decode canonical positive cell barcodes without accepting alternate spellings."""
    context = ReaderErrorContext("visium_hd", "cell barcodes", version)
    valid = names.str.fullmatch(r"cellid_[0-9]{9}-1", na=False)
    if not valid.all():
        raise ReaderFormatError(
            context=context,
            path=path,
            found="noncanonical cell barcode",
            expected="cellid_{positive_id:09d}-1",
        )

    values = names.str.slice(7, 16).to_numpy(dtype=np.uint32)
    if np.any(values == 0):
        raise ReaderFormatError(context=context, path=path, found="zero cell ID", expected="positive cell IDs")
    return values


def boundary_cell_ids(frame: pd.DataFrame, path: Path, *, context: ReaderErrorContext) -> NDArray[np.uint32]:
    """Require one unique, positive vendor cell ID per boundary feature."""
    expected = f"unique nonnull integer cell_id properties from 1 through {MAX_CELL_ID}"
    if "cell_id" not in frame:
        raise ReaderFormatError(context=context, path=path, found="missing cell_id", expected=expected)
    ids = frame["cell_id"]
    if ids.isna().any() or not pd.api.types.is_integer_dtype(ids.dtype):
        raise ReaderFormatError(context=context, path=path, found="null or non-integer cell_id", expected=expected)
    if ((ids <= 0) | (ids > MAX_CELL_ID)).any() or ids.duplicated().any():
        raise ReaderFormatError(
            context=context, path=path, found="duplicate or out-of-range cell_id", expected=expected
        )
    return ids.to_numpy(dtype=np.uint32)


def boundary_centroids(
    frame: pd.DataFrame, name: str, path: Path, *, context: ReaderErrorContext
) -> NDArray[np.float64] | None:
    """Return optional vendor centroid properties as one finite XY pair per feature."""
    if name not in frame:
        return None
    expected = "one finite [x, y] centroid per polygon"
    try:
        centroids = np.array(frame[name].tolist(), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ReaderFormatError(
            context=context, path=path, found="malformed centroid property", expected=expected
        ) from error
    if centroids.shape != (len(frame), 2) or not np.isfinite(centroids).all():
        raise ReaderFormatError(
            context=context, path=path, found="missing or nonfinite centroid property", expected=expected
        )
    return centroids


def validate_schema(schema: pa.Schema, required: pa.Schema, path: Path, *, context: ReaderErrorContext) -> None:
    """Require unique names and supported named fields; extra fields are allowed."""
    if len(set(schema.names)) != len(schema.names):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="duplicate column names",
            expected="unique column names",
        )

    for field in required:
        if field.name not in schema.names or schema.field(field.name).type != field.type:
            raise ReaderFormatError(
                context=context,
                path=path,
                found=str(schema),
                expected=f"column {field.name!r} with type {field.type}",
            )


def parse_bin_barcodes(
    names: pd.Index[str], size_um: int, shape: tuple[int, int], path: Path, *, version: str | None
) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    """Parse exact vendor barcodes without changing source observation order."""
    context = ReaderErrorContext("visium_hd", "matrix barcodes", version)
    pattern = re.compile(rf"s_{size_um:03d}um_([0-9]{{5}})_([0-9]{{5}})-1")
    rows = np.empty(len(names), dtype=np.uint32)
    cols = np.empty(len(names), dtype=np.uint32)
    for i, name in enumerate(names):
        match = pattern.fullmatch(name)
        if match is None:
            raise ReaderFormatError(
                context=context, path=path, found=repr(name), expected=f"s_{size_um:03d}um_RRRRR_CCCCC-1"
            )

        row, col = int(match[1]), int(match[2])
        if row >= shape[0] or col >= shape[1]:
            raise ReaderFormatError(
                context=context, path=path, found=repr(name), expected=f"indices within grid {shape}"
            )
        rows[i], cols[i] = row, col

    return rows, cols


def validate_position_batch(batch: pa.RecordBatch, path: Path, *, context: ReaderErrorContext) -> None:
    """Validate nonnull source values and finite pixel coordinates."""
    if any(column.null_count for column in batch.columns):
        raise ReaderFormatError(context=context, path=path, found="null positions", expected="nonnull position fields")

    tissue = batch.column("in_tissue").to_numpy()
    if np.any(tissue > 1):
        raise ReaderFormatError(
            context=context, path=path, found="in_tissue outside {0, 1}", expected="binary tissue flags"
        )

    for name in ("pxl_col_in_fullres", "pxl_row_in_fullres"):
        if not np.isfinite(batch.column(name).to_numpy()).all():
            raise ReaderFormatError(
                context=context, path=path, found=f"nonfinite {name}", expected="finite pixel coordinates"
            )


def validate_mappings(batch: pa.RecordBatch, path: Path, *, context: ReaderErrorContext) -> None:
    """Check mapping nullability and nucleus/cell membership relationships."""
    for field in MAPPINGS_SCHEMA:
        if not field.nullable and batch.column(field.name).null_count:
            raise ReaderFormatError(
                context=context, path=path, found=f"null {field.name}", expected="nonnull bin and membership fields"
            )

    in_cell = batch.column("in_cell").to_numpy(zero_copy_only=False)
    in_nucleus = batch.column("in_nucleus").to_numpy(zero_copy_only=False)
    if np.any(batch.column("cell_id").is_valid().to_numpy(zero_copy_only=False) != in_cell):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="cell_id nullability contradicts in_cell",
            expected="cell_id present exactly for in_cell bins",
        )
    if np.any(in_nucleus & ~in_cell):
        raise ReaderFormatError(
            context=context, path=path, found="nucleus bin outside a cell", expected="in_nucleus implies in_cell"
        )
