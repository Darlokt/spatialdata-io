"""Exact clustering identities survive complete SpatialData assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata.tests.helpers import assert_equal
from scipy import sparse
from spatialdata import read_zarr
from spatialdata.models import get_table_keys

from spatialdata_io import stereoseq
from tests.integration.readers.stereoseq._fixtures import write_analysis

if TYPE_CHECKING:
    from pathlib import Path


class TestStereoseqAnalysisResolution:
    """Large and precise decimals remain exact and valid element names."""

    @pytest.mark.parametrize("resolution", ["1e70", "1e-80", "0.123456789012345678901234567890123"])
    def test_extreme_exact_resolution_names_compose_in_public_spatialdata(
        self, tmp_path: Path, resolution: str
    ) -> None:
        path = write_analysis(tmp_path / f"sample.cellbin_{resolution}.h5ad")
        result = stereoseq(tmp_path, analysis_files=(path,))
        assert len(result.tables) == len(result.points) == 1
        table = next(iter(result.tables.values()))
        assert table.obs_names.tolist() == [str((10 << 32) | 20), str((30 << 32) | 40)]
        assert isinstance(table.X, sparse.sparray)

    @pytest.mark.parametrize("storage", ["h5ad", "zarr"])
    def test_inherited_analysis_slots(self, tmp_path: Path, storage: str) -> None:
        path = write_analysis(tmp_path / "sample.cellbin_0.5.h5ad")
        source = ad.read_h5ad(path)
        source.raw = ad.AnnData(
            sparse.csr_matrix([[1, 0, 2], [0, 3, 4]]),
            obs=pd.DataFrame(index=source.obs_names),
            var=pd.DataFrame({"source": ["z", "a", "b"]}, index=["raw-z", "raw-a", "raw-b"]),
        )
        source.layers["counts"] = sparse.csc_matrix([[1, 2], [3, 4]])
        source.layers["log1p"] = np.array([[0.25, -1.0], [2.5, 0.0]])
        source.layers["exon"] = sparse.csc_matrix([[0, 2], [1, 0]])
        source.layers["metadata"] = np.array([[1, 2], [3, 4]], dtype=np.int16)
        source.layers["sparse_metadata"] = sparse.csc_matrix([[5, 6], [7, 8]])
        source.varm["inherited"] = np.array([[10], [20]], dtype=np.int16)
        source.obsp["inherited"] = sparse.csr_matrix([[0, 1], [2, 0]])
        source.varp["inherited"] = sparse.csr_matrix([[0, 3], [4, 0]])
        source.uns["inherited"] = {"nested": {"note": "retained"}}
        source.write_h5ad(path)
        result = stereoseq(tmp_path, analysis_files=(path,))
        table = next(iter(result.tables.values()))
        destination = tmp_path / f"inherited.{storage}"
        if storage == "h5ad":
            table.write_h5ad(destination)
            restored = ad.read_h5ad(destination)
        else:
            result.write(destination)
            restored = next(iter(read_zarr(destination).tables.values()))

        assert isinstance(table.X, sparse.csr_array)
        assert all(isinstance(table.layers[name], sparse.csr_array) for name in ("counts", "log1p", "exon"))
        assert isinstance(table.layers["sparse_metadata"], sparse.csc_matrix)
        for actual in (table, restored):
            assert actual.obs_names.equals(source.obs_names)
            assert actual.var_names.equals(source.var_names)
            assert_equal(actual.raw, source.raw, exact=True)
            for name in ("counts", "log1p", "exon", "metadata", "sparse_metadata"):
                assert_equal(actual.layers[name], source.layers[name], exact=True)
            np.testing.assert_array_equal(actual.obsm["spatial"], source.obsm["spatial"])
            np.testing.assert_array_equal(actual.obsm["X_pca"], source.obsm["X_pca"])
            np.testing.assert_array_equal(actual.varm["inherited"], source.varm["inherited"])
            for name in ("obsp", "varp"):
                assert_equal(getattr(actual, name)["inherited"], getattr(source, name)["inherited"], exact=True)
            assert actual.uns["inherited"] == source.uns["inherited"]
            assert get_table_keys(actual) == get_table_keys(table)

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("location", ["dense", "sparse", "counts"])
    def test_nonfinite_expression_fails_at_final_normalization(
        self, tmp_path: Path, value: float, location: str
    ) -> None:
        path = write_analysis(tmp_path / "sample.cellbin_0.5.h5ad")
        table = ad.read_h5ad(path)
        values = np.array([[1.0, value], [2.0, 3.0]])
        if location == "counts":
            table.layers["counts"] = values
        else:
            table.X = values if location == "dense" else sparse.csr_matrix(values)
        table.write_h5ad(path)
        with pytest.raises(ValueError, match="finite values") as caught:
            stereoseq(tmp_path, analysis_files=(path,))
        assert any(str(path) in note and "stereoseq" in note for note in caught.value.__notes__)
