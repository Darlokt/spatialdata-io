"""Sparse bin matrices and bounded grid-index position alignment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers.visium_hd._metadata import read_bin_scale_factors
from spatialdata_io.readers.visium_hd._schemas import (
    POSITIONS_SCHEMA,
    parse_bin_barcodes,
    validate_position_batch,
    validate_schema,
)
from spatialdata_io.readers.visium_hd._tables import read_counts
from spatialdata_io.readers.visium_hd._transforms import bin_centers

if TYPE_CHECKING:
    from collections.abc import Iterator

    from anndata import AnnData
    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.visium_hd._layout import BinLayout
    from spatialdata_io.readers.visium_hd._transforms import SlideFrame

POSITION_BATCH_SIZE = 131_072


def read_grid_shape(positions: ArtifactFile) -> tuple[int, int]:
    """Infer grid bounds from exact Parquet maxima, with a projected fallback.

    Footer statistics avoid a pixel-position scan when the optional feature
    slice is absent. The consuming iterator independently proves complete,
    nonnull, unique coverage inside these bounds.
    """
    try:
        source = pq.ParquetFile(positions)
    except pa.ArrowInvalid as error:
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "tissue positions"),
            path=positions,
            found="malformed Parquet footer",
            expected="a readable positions Parquet artifact",
        ) from error

    with source:
        validate_schema(source.schema_arrow, POSITIONS_SCHEMA, positions, "tissue positions")
        if source.metadata.num_rows == 0:
            raise ReaderFormatError(
                context=ReaderErrorContext("visium_hd", "tissue positions"),
                path=positions,
                found="zero position rows",
                expected="a nonempty rectangular grid",
            )

        indices = [
            next(i for i in range(source.metadata.num_columns) if source.schema.column(i).path == name)
            for name in ("array_row", "array_col")
        ]
        maxima = [-1, -1]
        have_statistics = True
        for i in range(source.metadata.num_row_groups):
            group = source.metadata.row_group(i)
            for axis, column in enumerate(indices):
                statistics = group.column(column).statistics
                if statistics is None or not statistics.has_min_max:
                    have_statistics = False
                    break
                maxima[axis] = max(maxima[axis], int(statistics.max))

        if not have_statistics:
            maxima = [-1, -1]
            for batch in source.iter_batches(batch_size=POSITION_BATCH_SIZE, columns=["array_row", "array_col"]):
                for axis, name in enumerate(("array_row", "array_col")):
                    column = batch.column(name)
                    if column.null_count:
                        raise ReaderFormatError(
                            context=ReaderErrorContext("visium_hd", "tissue positions"),
                            path=positions,
                            found=f"null {name}",
                            expected="nonnull grid indices",
                        )
                    maxima[axis] = max(maxima[axis], int(column.to_numpy().max(initial=0)))

        return maxima[0] + 1, maxima[1] + 1


def iter_position_batches(positions: ArtifactFile, shape: tuple[int, int]) -> Iterator[pa.RecordBatch]:
    """Yield projected source batches, validating complete unique grid coverage.

    One boolean per grid site records coverage; decoded source columns have at
    most ``POSITION_BATCH_SIZE`` rows. No complete positions frame is built.
    """
    context = ReaderErrorContext("visium_hd", "tissue positions")
    try:
        source = pq.ParquetFile(positions)
    except pa.ArrowInvalid as error:
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "tissue positions"),
            path=positions,
            found="malformed Parquet footer",
            expected="a readable positions Parquet artifact",
        ) from error

    with source:
        validate_schema(source.schema_arrow, POSITIONS_SCHEMA, positions, "tissue positions")
        if source.metadata.num_rows != shape[0] * shape[1]:
            raise ReaderFormatError(
                context=context,
                path=positions,
                found=f"{source.metadata.num_rows} rows",
                expected=f"complete grid {shape}",
            )

        seen = np.zeros(shape[0] * shape[1], dtype=np.bool_)
        for batch in source.iter_batches(batch_size=POSITION_BATCH_SIZE, columns=POSITIONS_SCHEMA.names):
            validate_position_batch(batch, positions)
            rows = batch.column("array_row").to_numpy()
            cols = batch.column("array_col").to_numpy()
            if np.any(rows >= shape[0]) or np.any(cols >= shape[1]):
                raise ReaderFormatError(
                    context=context, path=positions, found="out-of-grid indices", expected=f"indices within {shape}"
                )

            indices = rows.astype(np.int64) * shape[1] + cols
            if np.any(seen[indices]) or np.unique(indices).size != indices.size:
                raise ReaderFormatError(
                    context=context, path=positions, found="duplicate grid indices", expected="one row per grid site"
                )
            seen[indices] = True
            yield batch


def read_positions(
    positions: ArtifactFile,
    shape: tuple[int, int],
    names: pd.Index[str],
    size_um: int,
) -> pd.DataFrame:
    """Select matrix rows by encoded grid index and verify their source barcodes."""
    rows, cols = parse_bin_barcodes(names, size_um, shape, positions)
    indices = rows.astype(np.int64) * shape[1] + cols
    # Search spatially sorted IDs, then scatter values back into matrix observation order.
    order = np.argsort(indices)
    sorted_indices = indices[order]

    tissue = np.empty(len(names), dtype=np.uint8)
    x = np.empty(len(names), dtype=np.float64)
    y = np.empty(len(names), dtype=np.float64)

    for batch in iter_position_batches(positions, shape):
        grid = batch.column("array_row").to_numpy().astype(np.int64) * shape[1] + batch.column("array_col").to_numpy()
        matches = np.searchsorted(sorted_indices, grid)
        valid = matches < len(indices)
        if not valid.any():
            continue

        source_rows = np.flatnonzero(valid)
        source_rows = source_rows[sorted_indices[matches[source_rows]] == grid[source_rows]]
        destination = order[matches[source_rows]]

        source_names = batch.column("barcode").take(pa.array(source_rows)).to_pylist()
        if source_names != names[destination].tolist():
            raise ReaderFormatError(
                context=ReaderErrorContext("visium_hd", "tissue positions"),
                path=positions,
                found="barcode differs at selected grid index",
                expected="exact matrix/positions barcode agreement",
            )

        tissue[destination] = batch.column("in_tissue").to_numpy()[source_rows]
        x[destination] = batch.column("pxl_col_in_fullres").to_numpy()[source_rows]
        y[destination] = batch.column("pxl_row_in_fullres").to_numpy()[source_rows]

    result = pa.table(
        {"in_tissue": tissue, "array_row": rows, "array_col": cols, "pxl_row_in_fullres": y, "pxl_col_in_fullres": x}
    ).to_pandas(types_mapper=pd.ArrowDtype)
    result.index = names
    return result


def read_bin_table(
    layout: BinLayout, shape: tuple[int, int], frame: SlideFrame, *, make_var_names_unique: bool
) -> AnnData:
    """Read one selected raw/filtered matrix and its physical bin centres."""
    read_bin_scale_factors(layout.scale_factors, layout.size_um)
    table = read_counts(layout.counts)
    positions = read_positions(layout.positions, shape, table.obs_names, layout.size_um)
    table.obs = positions

    rows: NDArray[np.uint32] = positions["array_row"].to_numpy(dtype=np.uint32)
    cols: NDArray[np.uint32] = positions["array_col"].to_numpy(dtype=np.uint32)
    table.obsm["spatial"] = bin_centers(frame, rows, cols, float(layout.size_um))

    if make_var_names_unique:
        table.var_names_make_unique()

    return table
