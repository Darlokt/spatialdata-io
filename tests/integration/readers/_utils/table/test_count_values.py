"""Exact finalized and promoted expression persists in standard AnnData stores."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.table import (
    SparseDuplicateSummary,
    TableLinkage,
    finalize_count_arrays,
    parse_linked_table,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestCountPersistence:
    @pytest.mark.parametrize("storage", ["h5ad", "zarr"])
    def test_finalized_count_family_round_trip(self, tmp_path: Path, storage: str) -> None:
        x, blank = finalize_count_arrays(
            (sparse.csr_array([[2**24 + 1]], dtype=np.int64), sparse.csr_array([[3]], dtype=np.uint8)),
            context=ReaderErrorContext("test", "count family"),
        )
        table = ad.AnnData(x)
        table.obsm["blank"] = blank
        path = tmp_path / f"counts.{storage}"
        if storage == "h5ad":
            table.write_h5ad(path)
            restored = ad.read_h5ad(path)
        else:
            table.write_zarr(path)
            restored = ad.read_zarr(path)
        matrix, auxiliary = restored.X, restored.obsm["blank"]
        assert isinstance(matrix, (sparse.csr_array, sparse.csr_matrix))
        assert isinstance(auxiliary, (sparse.csr_array, sparse.csr_matrix))
        assert matrix.dtype == auxiliary.dtype == np.dtype(np.float64)
        assert int(matrix.data[0]) == 2**24 + 1
        np.testing.assert_array_equal(auxiliary.data, [3])

    @pytest.mark.parametrize("storage", ["h5ad", "zarr"])
    def test_imported_promotion_and_linkage_round_trip(self, tmp_path: Path, storage: str) -> None:
        source = sparse.csr_array(
            (np.array([100, 100], dtype=np.int8), np.array([0, 0]), np.array([0, 2])), shape=(1, 1)
        )
        table = ad.AnnData(
            source,
            obs=pd.DataFrame({"region": pd.Categorical(["cells"]), "id": [1]}, index=["cell1"]),
            var=pd.DataFrame(index=["gene1"]),
        )
        reports: list[SparseDuplicateSummary] = []
        parse_linked_table(
            table,
            TableLinkage(("cells",), "region", "id"),
            context=ReaderErrorContext("test", "imported table"),
            expected_instance_ids={"cells": pd.Index([1])},
            duplicate_sums=reports,
        )
        assert len(reports) == 1
        assert reports[0].combined_entries == 1
        path = tmp_path / f"imported.{storage}"
        if storage == "h5ad":
            table.write_h5ad(path)
            restored = ad.read_h5ad(path)
        else:
            table.write_zarr(path)
            restored = ad.read_zarr(path)
        matrix = restored.X
        assert isinstance(matrix, (sparse.csr_array, sparse.csr_matrix))
        assert matrix.dtype == np.dtype(np.float32)
        assert int(matrix.data[0]) == 200
        assert restored.uns["spatialdata_attrs"]["instance_key"] == "id"
        np.testing.assert_array_equal(source.data, [100, 100])
