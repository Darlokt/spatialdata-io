"""Load DBiT H5AD expression and align rows to barcode-grid positions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd

from spatialdata_io._positions import POSITIONS_KEY, PositionsMetadata
from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers._utils.table import normalize_owned_anndata
from spatialdata_io.readers.dbit._schemas import (
    BARCODE_LENGTH,
    GRID_DTYPE,
    _validate_barcode_prefixes,
    _validate_mapped_positions,
    _validate_table,
)

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers._utils.table import SparseDuplicateSummary
    from spatialdata_io.readers.dbit._barcodes import _Barcodes


def _read_table(
    path: ArtifactFile, barcodes: _Barcodes, *, duplicate_sums: list[SparseDuplicateSummary] | None = None
) -> ad.AnnData:
    """Return an owned table sorted by A then B, with exact int64 grid coordinates."""
    context = ReaderErrorContext("dbit", "expression table")
    try:
        table = ad.read_h5ad(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=context, path=path)
        raise

    _validate_table(table, path)
    if "spatial" in table.obsm:
        table.obsm["spatial_source"] = table.obsm.pop("spatial")

    # Export IDs encode B then A; output coordinates are A/x then B/y.
    names = table.obs_names.str.slice(0, 2 * BARCODE_LENGTH).str.upper()
    _validate_barcode_prefixes(names, path)

    a = pd.Series([barcodes.a.get(value) for value in names.str.slice(BARCODE_LENGTH)], dtype=GRID_DTYPE)
    b = pd.Series([barcodes.b.get(value) for value in names.str.slice(0, BARCODE_LENGTH)], dtype=GRID_DTYPE)
    _validate_mapped_positions(a, b, path)

    # The schema validates eager H5AD frames; AnnData also types backed Dataset2D.
    obs = cast("pd.DataFrame", table.obs)
    obs["array_A"], obs["array_B"] = a.array, b.array

    # Convert before row selection: AnnData's copy must never duplicate a dense
    # expression matrix merely to establish the final observation order.
    # read_h5ad is eager. as_csr_array checks its actual third-party container.
    try:
        normalize_owned_anndata(
            table,
            expression_layers=("counts",) if "counts" in table.layers else (),
            duplicate_sums=[] if duplicate_sums is None else duplicate_sums,
        )
    except (TypeError, ValueError) as error:
        add_reader_context(error, context=context, path=path)
        raise

    # AnnData slices all observation-aligned annotations together. Keep the
    # owned table when the source already has the required order.
    order = np.lexsort((b.to_numpy(), a.to_numpy()))
    if not np.array_equal(order, np.arange(table.n_obs)):
        table = table[order].copy()

    # Eager AnnData row selection preserves the schema-validated pandas frames.
    obs = cast("pd.DataFrame", table.obs)
    table.obsm["spatial_grid"] = obs[["array_A", "array_B"]].to_numpy(dtype=np.int64)
    positions: PositionsMetadata = {
        "schema_version": 1,
        "coordinates": {
            "spatial_grid": {
                "axes": ["x", "y"],
                "units": "grid",
                "frame": "barcode_grid",
                "position": "one-based barcode channel ordinals (A, B)",
            }
        },
    }
    table.uns[POSITIONS_KEY] = positions

    return table
