"""AnnData storage integration tests for expression normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy import sparse

from spatialdata_io import stereoseq
from spatialdata_io.experimental import iss
from spatialdata_io.readers._utils.table import normalize_owned_anndata
from tests.integration.readers.stereoseq._fixtures import write_analysis

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pytest_mock import MockerFixture


def _object_index(values: list[str]) -> pd.Index:
    """Keep H5AD fixtures writable across the supported AnnData range."""
    return pd.Index(np.asarray(values, dtype=object), dtype=object)


class TestNormalizeOwnedAnndata:
    """Verify in-memory and backed H5AD boundaries."""

    def test_normalizes_in_memory_h5ad_without_reordering(self, tmp_path: Path) -> None:
        path = tmp_path / "table.h5ad"
        expected_x = sparse.csr_matrix([[0, 2], [3, 0]], dtype=np.int16)
        expected_counts = sparse.csc_matrix([[4, 0], [5, 6]], dtype=np.int32)
        obs_index = _object_index(["o2", "o1"])
        var_index = _object_index(["v2", "v1"])
        source = ad.AnnData(
            expected_x,
            obs=pd.DataFrame(
                {"sample": pd.Series(np.asarray(["a", "b"], dtype=object), index=obs_index, dtype=object)},
                index=obs_index,
            ),
            var=pd.DataFrame(
                {"symbol": pd.Series(np.asarray(["y", "x"], dtype=object), index=var_index, dtype=object)},
                index=var_index,
            ),
            layers={"counts": expected_counts},
        )
        source.write_h5ad(path)

        table = ad.read_h5ad(path)
        result = normalize_owned_anndata(table, expression_layers=("counts",))

        assert result is table
        assert isinstance(result.X, sparse.csr_array)
        assert isinstance(result.layers["counts"], sparse.csr_array)
        assert not isinstance(result.X, sparse.spmatrix)
        assert not isinstance(result.layers["counts"], sparse.spmatrix)
        assert result.X.has_canonical_format
        assert result.layers["counts"].has_canonical_format
        np.testing.assert_array_equal(result.X.toarray(), expected_x.toarray())
        np.testing.assert_array_equal(result.layers["counts"].toarray(), expected_counts.toarray())
        assert result.obs_names.tolist() == ["o2", "o1"]
        assert result.var_names.tolist() == ["v2", "v1"]
        assert result.obs["sample"].tolist() == ["a", "b"]
        assert result.var["symbol"].tolist() == ["y", "x"]

    def test_rejects_backed_h5ad_before_accessing_expression(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        path = tmp_path / "backed.h5ad"
        source = ad.AnnData(
            np.asarray([[1, 0], [0, 2]], dtype=np.float32),
            obs=pd.DataFrame(index=_object_index(["o1", "o2"])),
            var=pd.DataFrame(index=_object_index(["v1", "v2"])),
        )
        source.write_h5ad(path)
        table = ad.read_h5ad(path, backed="r")
        conversion = mocker.patch("spatialdata_io.readers._utils.table._normalize.as_csr_array")
        try:
            with pytest.raises(TypeError, match="not a backed table"):
                normalize_owned_anndata(table)
        finally:
            table.file.close()

        conversion.assert_not_called()


class TestCompressedSparseIntegrity:
    """Reject malformed artifacts in normalization and concrete public readers."""

    @pytest.mark.parametrize("factory", [sparse.csr_matrix, sparse.csc_matrix])
    @pytest.mark.parametrize("slot", ["X", "counts"])
    def test_h5ad_normalization_rejects_bad_indices(
        self,
        tmp_path: Path,
        factory: Callable[[np.ndarray], sparse.csr_matrix | sparse.csc_matrix],
        slot: str,
    ) -> None:
        source = ad.AnnData(sparse.csr_array(np.eye(2, dtype=np.int16)))
        expression = factory(np.eye(2, dtype=np.int16))
        if slot == "X":
            source.X = expression
        else:
            source.layers[slot] = expression
        path = tmp_path / "expression.h5ad"
        source.write_h5ad(path)
        with h5py.File(path, "r+") as handle:
            handle[f"{slot if slot == 'X' else 'layers/' + slot}/indices"][0] = 2
        table = ad.read_h5ad(path)

        with pytest.raises(ValueError, match="indices must be <") as error:
            normalize_owned_anndata(table, expression_layers=() if slot == "X" else (slot,))

        assert any(slot in note for note in error.value.__notes__)

    def test_stereoseq_rejects_malformed_analysis(self, tmp_path: Path) -> None:
        path = write_analysis(tmp_path / "sample.cellbin_0.5.h5ad")
        with h5py.File(path, "r+") as handle:
            handle["X/indices"][0] = 2

        with pytest.raises(ValueError, match="indices must be <"):
            stereoseq(tmp_path, analysis_files=(path,))

    def test_iss_rejects_malformed_expression(self, tmp_path: Path) -> None:
        tifffile.imwrite(
            tmp_path / "raw.tiff",
            np.zeros((2, 4, 4), dtype=np.uint16),
            photometric="minisblack",
            metadata={"axes": "CYX"},
        )
        tifffile.imwrite(tmp_path / "labels.tiff", np.ones((4, 4), dtype=np.uint16), photometric="minisblack")
        source = ad.AnnData(
            sparse.csr_array(np.ones((2, 3), dtype=np.int16)),
            obs=pd.DataFrame(index=["7", "2"]),
            var=pd.DataFrame(index=["a", "b", "c"]),
        )
        path = tmp_path / "cells.h5ad"
        source.write_h5ad(path)
        with h5py.File(path, "r+") as handle:
            handle["X/indices"][0] = 3

        with pytest.raises(ValueError, match="indices must be <"):
            iss(tmp_path, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad")
