"""Tests for the narrow sparse table package surface."""

from __future__ import annotations

from spatialdata_io.readers._utils import table


class TestTablePackage:
    """Keep reader-facing exports explicit and model-independent."""

    def test_exports_only_sparse_array_construction(self) -> None:
        assert table.__all__ == ["as_csr_array", "csr_array_from_triplets"]
