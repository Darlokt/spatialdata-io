"""Sparse table representation utilities for SpatialData readers.

The package owns representation and construction invariants only. Reader code
continues to own feature selection, identifiers, schema interpretation,
aggregation meaning, and SpatialData model construction.
"""

from __future__ import annotations

from spatialdata_io.readers._utils.table._arrays import as_csr_array, csr_array_from_triplets
from spatialdata_io.readers._utils.table._linkage import TableLinkage, parse_linked_table
from spatialdata_io.readers._utils.table._normalize import normalize_owned_anndata

__all__ = [
    "TableLinkage",
    "as_csr_array",
    "csr_array_from_triplets",
    "normalize_owned_anndata",
    "parse_linked_table",
]
