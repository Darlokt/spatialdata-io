"""Named source contracts for positions and segmentation mappings."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
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
MAPPINGS_TYPES = {"square_002um": pa.string(), "cell_id": pa.string(), "in_cell": pa.bool_(), "in_nucleus": pa.bool_()}


def validate_schema(schema: pa.Schema, required: pa.Schema, path: Path, artifact: str) -> None:
    """Require unique names and supported named fields; extra fields are allowed."""
    if len(set(schema.names)) != len(schema.names):
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", artifact),
            path=path,
            found="duplicate column names",
            expected="unique column names",
        )

    for field in required:
        if field.name not in schema.names or schema.field(field.name).type != field.type:
            raise ReaderFormatError(
                context=ReaderErrorContext("visium_hd", artifact),
                path=path,
                found=str(schema),
                expected=f"column {field.name!r} with type {field.type}",
            )


def parse_bin_barcodes(
    names: pd.Index[str], size_um: int, shape: tuple[int, int], path: Path
) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    """Parse exact vendor barcodes without changing source observation order."""
    context = ReaderErrorContext("visium_hd", "matrix barcodes")
    if not names.is_unique:
        raise ReaderFormatError(context=context, path=path, found="duplicate barcodes", expected="unique barcodes")

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


def validate_position_batch(batch: pa.RecordBatch, path: Path) -> None:
    """Validate nonnull source values and finite pixel coordinates."""
    context = ReaderErrorContext("visium_hd", "tissue positions")
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


def validate_mapping_schema(schema: pa.Schema, path: Path) -> None:
    """Require projected 2 µm-to-cell mapping fields with their source types."""
    validate_schema(
        schema, pa.schema([pa.field(name, dtype) for name, dtype in MAPPINGS_TYPES.items()]), path, "barcode mappings"
    )


def validate_mappings(batch: pa.RecordBatch | pa.Table, path: Path) -> None:
    """Check mapping nullability and nucleus/cell membership relationships."""
    context = ReaderErrorContext("visium_hd", "barcode mappings")
    for name in ("square_002um", "in_cell", "in_nucleus"):
        if batch.column(name).null_count:
            raise ReaderFormatError(
                context=context, path=path, found=f"null {name}", expected="nonnull bin and membership fields"
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
