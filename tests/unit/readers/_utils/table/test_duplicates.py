"""Imported duplicate coordinates preserve values through exact promotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from spatialdata_io.readers._utils.table import SparseDuplicateSummary, normalize_owned_anndata
from spatialdata_io.readers._utils.table._duplicates import normalize_sparse_duplicates
from spatialdata_io.readers._utils.table._normalize import _normalized_expression

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytest_mock import MockerFixture


def _source(values: Sequence[int | float], dtype: str, storage: str) -> sparse.csr_array | sparse.csc_array:
    buffers = (np.asarray(values, dtype=dtype), np.zeros(len(values), dtype=np.int32), np.array([0, len(values)]))
    return sparse.csr_array(buffers, shape=(1, 2)) if storage == "csr" else sparse.csc_array(buffers, shape=(2, 1))


class TestNormalizeSparseDuplicates:
    @pytest.mark.parametrize("storage", ["csr", "csc"])
    @pytest.mark.parametrize(
        ("dtype", "values", "expected_dtype"),
        [
            ("int8", [100, 100], "int32"),
            ("int8", [-100, -100], "int32"),
            ("uint8", [150, 150], "uint32"),
            ("int16", [30000, 30000], "int32"),
            ("uint16", [65535, 1], "uint32"),
            ("int32", [2**31 - 1, 1], "int64"),
            ("uint32", [2**32 - 1, 1], "uint64"),
            ("int64", [2**63 - 1, 1], "uint64"),
            ("int64", [-(2**63), -(2**63)], "float64"),
            ("uint64", [2**63 + 1, 2**63 - 1], "float64"),
            ("int8", [100, 100, -100], "int8"),
            ("int16", [100, 100], "int16"),
            ("int8", [100, -100], "int8"),
        ],
    )
    def test_exact_sums_and_required_promotion(
        self, storage: str, dtype: str, values: list[int], expected_dtype: str
    ) -> None:
        source = _source(values, dtype, storage)
        before = source.data.copy(), source.indices.copy(), source.indptr.copy()
        result = normalize_sparse_duplicates(source)
        assert result.dtype == np.dtype(expected_dtype)
        assert result.has_canonical_format
        assert result.nnz == 1
        assert int(result.data[0]) == sum(values)
        assert isinstance(result, sparse.csr_array)
        for actual, original in zip((source.data, source.indices, source.indptr), before, strict=True):
            np.testing.assert_array_equal(actual, original)

    def test_inexact_fallback_rejected(self) -> None:
        source = _source([2**64 - 1, 2**64 - 1], "uint64", "csr")
        with pytest.raises(ValueError, match="no supported lossless"):
            normalize_sparse_duplicates(source)

    def test_fallback_must_preserve_unaffected_entries(self) -> None:
        source = sparse.csr_array(
            (np.array([2**63, 2**63, 2**53 + 1], dtype=np.uint64), np.array([0, 0, 1]), np.array([0, 3])),
            shape=(1, 2),
        )
        with pytest.raises(ValueError, match="no supported lossless"):
            normalize_sparse_duplicates(source)

    @pytest.mark.parametrize("dtype", ["int8", "int64", "uint64", "float32", "float64", "bool"])
    def test_canonical_buffers_are_borrowed(self, dtype: str) -> None:
        source = sparse.csr_array([[1, 0, 2]], dtype=dtype)
        result = normalize_sparse_duplicates(source)
        assert result.dtype == source.dtype
        assert result is source
        assert np.shares_memory(result.data, source.data)
        assert np.shares_memory(result.indices, source.indices)

    def test_cached_canonical_flag_cannot_hide_overflow(self) -> None:
        source = _source([100, 100], "int8", "csr")
        source.has_canonical_format = True
        source.has_sorted_indices = True
        result = normalize_sparse_duplicates(source)
        assert int(result.data[0]) == 200
        assert result.dtype == np.dtype(np.int32)

    def test_sparse_structure_validated_before_conversion(self) -> None:
        source = _source([100, 100], "int8", "csr")
        source.indices[0] = 2
        with pytest.raises(ValueError, match="active indices"):
            normalize_sparse_duplicates(source)

    def test_unsorted_unique_entries_are_not_summed(self) -> None:
        source = sparse.csr_array(
            (np.array([127, 126], dtype=np.int8), np.array([1, 0]), np.array([0, 2])), shape=(1, 2)
        )
        result = normalize_sparse_duplicates(source)
        assert result.dtype == source.dtype
        np.testing.assert_array_equal(result.data, [126, 127])
        assert result.nnz == source.nnz

    @pytest.mark.parametrize("storage", ["csr", "csc"])
    def test_float_and_boolean_semantics_remain(self, storage: str) -> None:
        result = normalize_sparse_duplicates(_source([0.5, -0.25], "float32", storage))
        assert result.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(result.data, [0.25])
        result = normalize_sparse_duplicates(_source([1, 1], "bool", storage))
        assert result.dtype == np.dtype(bool)
        np.testing.assert_array_equal(result.data, [True])

    def test_never_densifies(self, mocker: MockerFixture) -> None:
        source = _source([100, 100], "int8", "csr")
        mocker.patch.object(source, "toarray", side_effect=AssertionError("densification"))
        result = normalize_sparse_duplicates(source)
        assert int(result.data[0]) == 200

    def test_empty_and_legacy_compressed_containers(self) -> None:
        assert normalize_sparse_duplicates(sparse.csr_array((0, 2), dtype=np.int8)).shape == (0, 2)
        for source in (
            sparse.csr_matrix(_source([100, 100], "int8", "csr")),
            sparse.csc_matrix(_source([100, 100], "int8", "csc")),
        ):
            assert int(normalize_sparse_duplicates(source).data[0]) == 200


class TestSparseDuplicateSummary:
    def test_summary_reports_promotion_and_safe_sums(self) -> None:
        promoted = SparseDuplicateSummary("X", 1, np.dtype(np.int8), np.dtype(np.int32))
        assert str(promoted) == "combined 1 duplicate sparse entry in X; promoted int8 to int32 to preserve exact sums"
        safe = SparseDuplicateSummary("layers['counts']", 2, np.dtype(np.int16), np.dtype(np.int16))
        assert str(safe) == "combined 2 duplicate sparse entries in layers['counts']"


class TestImportedNormalization:
    def test_collects_multiple_slots_without_emitting_warning(self, recwarn: pytest.WarningsRecorder) -> None:
        table = ad.AnnData(_source([100, 100], "int8", "csr"))
        table.layers["counts"] = _source([150, 150], "uint8", "csr")
        reports: list[SparseDuplicateSummary] = []
        assert normalize_owned_anndata(table, expression_layers=("counts",), duplicate_sums=reports) is table
        assert [report.slot for report in reports] == ["X", "layers['counts']"]
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.dtype == np.dtype(np.int32)
        assert table.layers["counts"].dtype == np.dtype(np.uint32)
        assert len(recwarn) == 0
        normalize_owned_anndata(table, expression_layers=("counts",), duplicate_sums=reports)
        assert len(reports) == 2

    def test_does_not_report_sorting_only(self) -> None:
        source = sparse.csr_array((np.array([2, 1]), np.array([1, 0]), np.array([0, 2])), shape=(1, 2))
        reports: list[SparseDuplicateSummary] = []
        normalize_owned_anndata(ad.AnnData(source), duplicate_sums=reports)
        assert reports == []

    def test_rejects_noncompressed_imported_sparse_storage(self) -> None:
        with pytest.raises(TypeError, match="CSR or CSC"):
            _normalized_expression(sparse.coo_array([[1]]), location="X", duplicate_sums=[])
