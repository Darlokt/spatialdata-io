"""Read the DBiT H5AD table and place every observation on the barcode grid.

The expression table is an AnnData H5AD file whose observation names encode the
spot's two barcodes, B first::

    GGGGTTTTAACCGGTT-1
    └──B───┘└──A───┘└ optional suffix, kept as part of the identifier

Both halves are looked up case-insensitively in the barcode map. A is the x axis
and B the y axis; the one-based ordinals are written to ``obs["array_A"]``,
``obs["array_B"]`` and ``obsm["spatial_grid"]`` (int64, columns A then B), and
``uns["spatialdata_io_positions"]`` declares ``spatial_grid`` as one-based
ordinals of the ``grid_pixels`` frame. Imported ``obs`` and ``var`` column
names that a SpatialData table does not accept (``my note``) are sanitized with
one warning per frame before these columns are added. An imported ``obsm["spatial"]`` is kept unchanged as
``obsm["spatial_source"]``; no physical ``spatial`` embedding is produced.

The table may hold any subset of the grid. Rows stay in source order here: the
reader links the table first, which converts dense expression to sparse CSR,
and only then sorts the rows by A and then B (see ``sort_rows``). Sorting
before that conversion would copy a dense matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd

from spatialdata_io._positions import POSITIONS_KEY, PositionsMetadata
from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers._utils.names import sanitize_column_names
from spatialdata_io.readers.dbit._schemas import (
    BARCODE_LENGTH,
    GRID_DTYPE,
    GRID_FRAME,
    validate_barcode_prefixes,
    validate_mapped_positions,
    validate_table,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.dbit._barcodes import Barcodes

_CONTEXT = ReaderErrorContext("dbit", "expression table")
# Observation columns the reader writes; imported columns that equal one
# ignoring letter case are renamed, and exact ones are replaced as before.
READER_OBS_COLUMNS = ("array_A", "array_B", "region", "sample", "pixel_id")


def read_table(path: ArtifactFile, barcodes: Barcodes) -> ad.AnnData:
    """Return the owned source-order table with exact A/B grid positions.

    Expression storage is left as read; final linkage normalizes it.
    """
    try:
        table = ad.read_h5ad(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=_CONTEXT, path=path)
        raise

    validate_table(table, path)
    # validate_table proved that read_h5ad returned eager DataFrames.
    obs, var = cast("pd.DataFrame", table.obs), cast("pd.DataFrame", table.var)
    obs.columns = sanitize_column_names(list(obs.columns), source="dbit: the H5AD obs", reserved=READER_OBS_COLUMNS)
    var.columns = sanitize_column_names(list(var.columns), source="dbit: the H5AD var")
    if "spatial" in table.obsm:
        table.obsm["spatial_source"] = table.obsm.pop("spatial")

    names = table.obs_names.str.slice(0, 2 * BARCODE_LENGTH).str.upper()
    validate_barcode_prefixes(names, path)
    a = pd.Series([barcodes.a.get(value) for value in names.str.slice(BARCODE_LENGTH)], dtype=GRID_DTYPE)
    b = pd.Series([barcodes.b.get(value) for value in names.str.slice(0, BARCODE_LENGTH)], dtype=GRID_DTYPE)
    validate_mapped_positions(a, b, path)

    obs["array_A"], obs["array_B"] = a.array, b.array
    table.obsm["spatial_grid"] = np.column_stack((a.to_numpy(np.int64), b.to_numpy(np.int64)))
    positions: PositionsMetadata = {
        "schema_version": 1,
        "coordinates": {
            "spatial_grid": {
                "axes": ["x", "y"],
                "units": "grid",
                "frame": GRID_FRAME,
                "position": "one-based barcode channel ordinals (A, B)",
            }
        },
    }
    table.uns[POSITIONS_KEY] = positions
    return table


def grid_order(coordinates: npt.NDArray[np.int64]) -> npt.NDArray[np.intp]:
    """Return the row permutation that sorts ``(A, B)`` positions by A, then B."""
    return np.lexsort((coordinates[:, 1], coordinates[:, 0]))


def sort_rows(table: ad.AnnData, order: npt.NDArray[np.intp]) -> ad.AnnData:
    """Return ``table`` with its rows in ``order``; an already ordered table is not copied."""
    if np.array_equal(order, np.arange(table.n_obs)):
        return table
    # AnnData reorders every observation-aligned slot together.
    return table[order].copy()
