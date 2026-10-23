"""Load DBiT H5AD expression and align rows to barcode-grid positions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.table import as_csr_array

if TYPE_CHECKING:
    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.dbit._barcodes import _Barcodes

_BARCODE_LENGTH = 8


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

    if table.X is None or not table.n_obs or not table.n_vars:
        raise ReaderFormatError(
            context=context, path=path, found="empty or absent expression", expected="nonempty observation-by-feature X"
        )
    if (
        not table.obs_names.is_unique
        or not table.var_names.is_unique
        or table.var_names.hasnans
        or (table.var_names == "").any()
    ):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="invalid identifiers",
            expected="unique observations and nonempty unique features",
        )

    # Export IDs encode B then A, although output coordinates are A/x then B/y.
    # Ignore suffixes for lookup and uniqueness, preserving the original IDs.
    names = table.obs_names.str[: 2 * _BARCODE_LENGTH].str.upper()
    if not names.str.fullmatch(r"[ACGT]{16}").all() or not names.is_unique:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="malformed or duplicate barcode pairs",
            expected="one observation per 16-base B-then-A barcode prefix. Optional suffixes are retained",
        )

    a = pd.Series(names.str[_BARCODE_LENGTH:]).map(barcodes.a)
    b = pd.Series(names.str[:_BARCODE_LENGTH]).map(barcodes.b)
    if a.isna().any() or b.isna().any():
        raise ReaderFormatError(
            context=context,
            path=path,
            found="unknown observation barcodes",
            expected="every B/A barcode in the position map",
        )

    table.obs["array_A"] = a.to_numpy(dtype=np.int64)
    table.obs["array_B"] = b.to_numpy(dtype=np.int64)

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

    # The eager H5AD owner and eager row selection always produce pandas obs.
    obs = cast("pd.DataFrame", table.obs)
    table.obsm["spatial"] = obs[["array_A", "array_B"]].to_numpy(dtype=np.int64)

    # Any previous target belongs to the source container, not this new grid.
    table.uns.pop("spatialdata_attrs", None)
    return table
