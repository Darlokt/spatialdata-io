"""Load DBiT H5AD expression and align rows to barcode-grid positions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers._utils.table import as_csr_array
from spatialdata_io.readers.dbit._schemas import (
    BARCODE_LENGTH,
    GRID_DTYPE,
    _source_linkage_columns,
    _validate_barcode_prefixes,
    _validate_mapped_positions,
    _validate_output_column,
    _validate_table,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.dbit._barcodes import _Barcodes


def _expression_array(
    values: npt.NDArray[np.generic] | sparse.sparray | sparse.spmatrix,
) -> sparse.csr_array:
    """Canonicalize expression without overflowing duplicate integer entries."""
    if (
        isinstance(values, (sparse.csr_array, sparse.csr_matrix, sparse.csc_array, sparse.csc_matrix))
        and values.dtype.kind in "iu"
        and not values.has_canonical_format
    ):
        # H5AD compressed sparse buffers may contain duplicate entries. Their
        # sum must fit the declared source dtype. SciPy otherwise wraps silently.
        # Canonical input never enters this exact-integer fallback or allocates it.
        limits = np.iinfo(values.dtype)
        for start, stop in zip(values.indptr[:-1], values.indptr[1:], strict=True):
            totals: dict[int, int] = {}
            for index, value in zip(values.indices[start:stop], values.data[start:stop], strict=True):
                key = int(index)
                totals[key] = totals.get(key, 0) + int(value)

            if any(total < limits.min or total > limits.max for total in totals.values()):
                message = f"Duplicate expression entries overflow source dtype {values.dtype}. Use a wider count dtype."
                raise ValueError(message)

    return as_csr_array(values)


def _read_table(path: ArtifactFile, barcodes: _Barcodes) -> ad.AnnData:
    """Return an owned table sorted by A then B, with exact int64 grid coordinates."""
    context = ReaderErrorContext("dbit", "expression table")
    try:
        table = ad.read_h5ad(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=context, path=path)
        raise

    _validate_table(table, path)
    source_linkage_columns = _source_linkage_columns(table, path)

    # Export IDs encode B then A; output coordinates are A/x then B/y.
    names = table.obs_names.str.slice(0, 2 * BARCODE_LENGTH).str.upper()
    _validate_barcode_prefixes(names, path)

    a = pd.Series([barcodes.a.get(value) for value in names.str.slice(BARCODE_LENGTH)], dtype=GRID_DTYPE)
    b = pd.Series([barcodes.b.get(value) for value in names.str.slice(0, BARCODE_LENGTH)], dtype=GRID_DTYPE)
    _validate_mapped_positions(a, b, path)

    # The schema validates eager H5AD frames; AnnData also types backed Dataset2D.
    obs = cast("pd.DataFrame", table.obs)
    _validate_output_column(obs, "array_A", a, path=path)
    _validate_output_column(obs, "array_B", b, path=path)
    obs["array_A"], obs["array_B"] = a.array, b.array

    # Convert before row selection: AnnData's copy must never duplicate a dense
    # expression matrix merely to establish the final observation order.
    # read_h5ad is eager. as_csr_array checks its actual third-party container.
    try:
        table.X = _expression_array(cast("npt.NDArray[np.generic] | sparse.sparray | sparse.spmatrix", table.X))
        if "counts" in table.layers:
            table.layers["counts"] = _expression_array(
                cast("npt.NDArray[np.generic] | sparse.sparray | sparse.spmatrix", table.layers["counts"])
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
    table.obsm["spatial"] = obs[["array_A", "array_B"]].to_numpy(dtype=np.int64)

    # A declared old linkage identifies metadata that the new grid replaces.
    # Unlinked columns with these reserved names remain preserved annotations.
    table.uns.pop("spatialdata_attrs", None)
    for key in source_linkage_columns:
        del obs[key]

    return table
