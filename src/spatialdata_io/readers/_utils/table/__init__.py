"""Expression, annotation storage, and linkage utilities for SpatialData readers.

The package owns representation and construction invariants only. Reader code
continues to own feature selection, identifiers, schema interpretation,
aggregation meaning, and SpatialData model construction.
"""

from __future__ import annotations

from spatialdata_io.readers._utils.table._arrays import as_csr_array
from spatialdata_io.readers._utils.table._compat import prepare_anndata_dataframe
from spatialdata_io.readers._utils.table._count_values import count_accumulation_dtype, finalize_count_arrays
from spatialdata_io.readers._utils.table._duplicates import SparseDuplicateSummary, normalize_sparse_duplicates
from spatialdata_io.readers._utils.table._linkage import TableLinkage, parse_linked_table
from spatialdata_io.readers._utils.table._normalize import normalize_owned_anndata

__all__ = [
    "SparseDuplicateSummary",
    "TableLinkage",
    "as_csr_array",
    "count_accumulation_dtype",
    "finalize_count_arrays",
    "normalize_owned_anndata",
    "normalize_sparse_duplicates",
    "parse_linked_table",
    "prepare_anndata_dataframe",
]
