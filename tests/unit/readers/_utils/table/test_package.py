"""Tests for the narrow sparse table package surface."""

from __future__ import annotations

from spatialdata_io.readers._utils import table


class TestTablePackage:
    """Keep reader-facing exports explicit."""

    def test_exports_only_representation_boundaries(self) -> None:
        """Expose only intentional table representation and linkage operations."""
        assert table.__all__ == [
            "TableLinkage",
            "as_csr_array",
            "csr_array_from_triplets",
            "normalize_owned_anndata",
            "parse_linked_table",
            "prepare_anndata_dataframe",
        ]
