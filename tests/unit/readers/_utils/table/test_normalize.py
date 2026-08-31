"""Tests for normalization of reader-owned AnnData expression storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import PropertyMock

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import pytest
from dask import delayed
from scipy import sparse

from spatialdata_io.readers._utils.table import normalize_owned_anndata

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_mock import MockerFixture


def _assert_canonical_csr(data: object) -> sparse.csr_array:
    result = cast("sparse.csr_array", data)
    assert isinstance(result, sparse.csr_array)
    assert isinstance(result, sparse.sparray)
    assert not isinstance(result, sparse.spmatrix)
    assert result.has_canonical_format
    assert result.has_sorted_indices
    return result


def _identity_dense(values: np.ndarray) -> np.ndarray:
    return values


class TestNormalizeOwnedAnndata:
    """Validate eager normalization, ownership, and selected-layer behavior."""

    @pytest.mark.parametrize(
        "factory",
        [
            _identity_dense,
            sparse.csr_matrix,
            sparse.csc_matrix,
            sparse.csr_array,
            sparse.csc_array,
        ],
    )
    def test_normalizes_supported_anndata_expression_storage(
        self,
        factory: Callable[[np.ndarray], np.ndarray | sparse.sparray | sparse.spmatrix],
    ) -> None:
        values = np.asarray([[0, 2], [3, 0]], dtype=np.int16)
        adata = ad.AnnData(factory(values), obs=pd.DataFrame(index=["a", "b"]), var=pd.DataFrame(index=["x", "y"]))

        result = normalize_owned_anndata(adata)

        normalized = _assert_canonical_csr(result.X)
        assert result is adata
        assert normalized.dtype == values.dtype
        np.testing.assert_array_equal(normalized.toarray(), values)
        assert result.obs_names.tolist() == ["a", "b"]
        assert result.var_names.tolist() == ["x", "y"]

    @pytest.mark.parametrize("shape", [(0, 0), (0, 4), (3, 0)])
    def test_preserves_empty_shapes(self, shape: tuple[int, int]) -> None:
        adata = ad.AnnData(np.empty(shape, dtype=np.float32))

        normalize_owned_anndata(adata)

        normalized = _assert_canonical_csr(adata.X)
        assert normalized.shape == shape
        assert normalized.dtype == np.float32

    def test_allows_metadata_only_table(self) -> None:
        adata = ad.AnnData(shape=(2, 3), obs={"sample": ["a", "b"]})

        result = normalize_owned_anndata(adata)

        assert result is adata
        assert result.X is None
        assert result.shape == (2, 3)
        assert result.obs["sample"].tolist() == ["a", "b"]

    def test_normalizes_only_explicit_expression_layers(self) -> None:
        x = np.asarray([[1, 0], [0, 2]], dtype=np.int32)
        counts = sparse.csc_matrix([[3, 0], [4, 5]], dtype=np.int16)
        metadata = np.asarray([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        adata = ad.AnnData(x, layers={"counts": counts, "metadata": metadata})
        metadata_before = adata.layers["metadata"]
        layer_order = tuple(adata.layers)

        normalize_owned_anndata(adata, expression_layers=("counts",))

        _assert_canonical_csr(adata.X)
        normalized_counts = _assert_canonical_csr(adata.layers["counts"])
        np.testing.assert_array_equal(normalized_counts.toarray(), counts.toarray())
        assert adata.layers["metadata"] is metadata_before
        assert tuple(adata.layers) == layer_order

    def test_leaves_raw_and_annotations_unchanged(self) -> None:
        adata = ad.AnnData(
            np.asarray([[1, 0], [0, 2]], dtype=np.float32),
            obs=pd.DataFrame({"kind": ["a", "b"]}, index=["o1", "o2"]),
            var=pd.DataFrame({"symbol": ["x", "y"]}, index=["v1", "v2"]),
            uns={"source": {"version": 1}},
        )
        adata.obsm["spatial"] = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        adata.varm["loadings"] = np.asarray([[0.5], [1.5]])
        adata.obsp["distances"] = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
        adata.varp["correlations"] = sparse.csr_matrix([[1.0, 0.5], [0.5, 1.0]])
        raw_var = pd.DataFrame({"symbol": ["x", "y"]}, index=["v1", "v2"])
        adata.raw = ad.AnnData(np.asarray([[6, 0], [0, 7]], dtype=np.int16), var=raw_var)
        raw_x = adata.raw.X
        spatial = adata.obsm["spatial"]
        loadings = adata.varm["loadings"]
        distances = adata.obsp["distances"]
        correlations = adata.varp["correlations"]
        obs = adata.obs
        var = adata.var
        uns = adata.uns

        normalize_owned_anndata(adata)

        assert adata.raw.X is raw_x
        assert isinstance(adata.raw.X, np.ndarray)
        assert adata.obsm["spatial"] is spatial
        assert adata.varm["loadings"] is loadings
        assert adata.obsp["distances"] is distances
        assert adata.varp["correlations"] is correlations
        assert adata.obs is obs
        assert adata.var is var
        assert adata.uns is uns

    def test_reuses_canonical_csr_array(self) -> None:
        source = sparse.csr_array([[1, 0], [0, 2]], dtype=np.int32)
        adata = ad.AnnData(source)

        normalize_owned_anndata(adata)

        assert adata.X is source

    def test_canonical_csr_matrix_shares_storage(self) -> None:
        source = sparse.csr_matrix([[1, 0], [0, 2]], dtype=np.int32)
        adata = ad.AnnData(source)

        normalize_owned_anndata(adata)

        normalized = _assert_canonical_csr(adata.X)
        assert np.shares_memory(normalized.data, source.data)
        assert np.shares_memory(normalized.indices, source.indices)
        assert np.shares_memory(normalized.indptr, source.indptr)

    def test_noncanonical_csr_is_copied_before_normalization(self) -> None:
        source = sparse.csr_matrix(
            (
                np.asarray([1, 2, 3], dtype=np.int16),
                np.asarray([1, 0, 0], dtype=np.int32),
                np.asarray([0, 2, 3], dtype=np.int32),
            ),
            shape=(2, 2),
        )
        source_data = source.data.copy()
        source_indices = source.indices.copy()
        source_indptr = source.indptr.copy()
        adata = ad.AnnData(source)

        normalize_owned_anndata(adata)

        normalized = _assert_canonical_csr(adata.X)
        assert not np.shares_memory(normalized.data, source.data)
        np.testing.assert_array_equal(source.data, source_data)
        np.testing.assert_array_equal(source.indices, source_indices)
        np.testing.assert_array_equal(source.indptr, source_indptr)
        assert not source.has_canonical_format

    @pytest.mark.parametrize(
        ("expression_layers", "error", "match"),
        [
            (("counts", "counts"), ValueError, "duplicates"),
            (("counts", "missing"), KeyError, "missing"),
        ],
    )
    def test_validates_all_layer_names_before_mutation(
        self,
        expression_layers: tuple[str, ...],
        error: type[Exception],
        match: str,
    ) -> None:
        x = np.asarray([[1, 0], [0, 2]], dtype=np.int16)
        counts = np.asarray([[3, 0], [0, 4]], dtype=np.int16)
        adata = ad.AnnData(x, layers={"counts": counts})

        with pytest.raises(error, match=match):
            normalize_owned_anndata(adata, expression_layers=expression_layers)

        assert adata.X is x
        assert adata.layers["counts"] is counts

    def test_rejects_view_without_copying(self, mocker: MockerFixture) -> None:
        adata = ad.AnnData(np.asarray([[1, 0], [0, 2]], dtype=np.int16))
        view = adata[:1, :]
        copy = mocker.patch.object(ad.AnnData, "copy", side_effect=AssertionError("must not copy"))

        with pytest.raises(ValueError, match="not a view"):
            normalize_owned_anndata(view)

        copy.assert_not_called()
        assert view.is_view

    def test_rejects_lazy_x_without_computing(self) -> None:
        executions: list[bool] = []

        @delayed
        def make_values() -> np.ndarray:
            executions.append(True)
            return np.asarray([[1, 0], [0, 2]], dtype=np.float32)

        lazy = da.from_delayed(make_values(), shape=(2, 2), dtype=np.float32)
        adata = ad.AnnData(lazy)

        with pytest.raises(TypeError, match="plain NumPy ndarray"):
            normalize_owned_anndata(adata)

        assert executions == []
        assert adata.X is lazy

    @pytest.mark.parametrize(
        ("values", "error", "match"),
        [
            (np.asarray([[1]], dtype=np.float16), TypeError, "Float16"),
            (np.asarray([[1 + 2j]]), TypeError, "complex"),
            (np.asarray([[np.nan]], dtype=np.float32), ValueError, "finite"),
        ],
    )
    def test_rejects_invalid_expression_values(
        self,
        values: np.ndarray,
        error: type[Exception],
        match: str,
    ) -> None:
        adata = ad.AnnData(values)

        with pytest.raises(error, match=match) as exception:
            normalize_owned_anndata(adata)

        assert exception.value.__notes__ == ["while normalizing adata.X"]

    def test_rejects_higher_dimensional_x(self, mocker: MockerFixture) -> None:
        adata = ad.AnnData(shape=(2, 3))
        # AnnData versions differ in whether construction rejects this corrupt
        # state. Replace the property to exercise this boundary directly.
        mocker.patch.object(
            ad.AnnData,
            "X",
            new_callable=PropertyMock,
            return_value=np.ones((2, 3, 4), dtype=np.float32),
        )

        with pytest.raises(ValueError, match="two-dimensional"):
            normalize_owned_anndata(adata)

    def test_documents_partial_mutation_after_later_failure(self) -> None:
        x = np.asarray([[1, 0], [0, 2]], dtype=np.int16)
        invalid = np.ma.array([[3, 0], [0, 4]], mask=False)
        adata = ad.AnnData(x, layers={"invalid": invalid})

        with pytest.raises(TypeError, match="plain NumPy ndarray") as exception:
            normalize_owned_anndata(adata, expression_layers=("invalid",))

        _assert_canonical_csr(adata.X)
        assert isinstance(adata.layers["invalid"], np.ma.MaskedArray)
        assert exception.value.__notes__ == ["while normalizing adata.layers['invalid']"]
