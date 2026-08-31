"""Sparse table representation utilities for SpatialData readers.

The package owns representation and construction invariants only. Reader code
continues to own feature selection, identifiers, schema interpretation,
aggregation meaning, and SpatialData model construction.
"""

from __future__ import annotations

from spatialdata_io.readers._utils.table._arrays import as_csr_array, csr_array_from_triplets

__all__ = ["as_csr_array", "csr_array_from_triplets"]
