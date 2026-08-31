"""Tests for canonical sparse-array construction."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import dask.array as da
import numpy as np
import pytest
from dask import delayed
from scipy import sparse

from spatialdata_io.readers._utils.table import as_csr_array, csr_array_from_triplets

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_mock import MockerFixture


def _assert_canonical_csr(result: sparse.csr_array) -> None:
    assert sparse.issparse(result)
    assert result.format == "csr"
    assert isinstance(result, sparse.csr_array)
    assert isinstance(result, sparse.sparray)
    assert not isinstance(result, sparse.spmatrix)
    assert result.has_canonical_format
    assert result.has_sorted_indices


class TestAsCsrArray:
    """Validate supported conversions without densification or mutation."""

    @pytest.mark.parametrize(
        "data",
        [
            np.asarray([[True, False], [False, True]], dtype=np.bool_),
            np.asarray([[1, 0], [-2, 3]], dtype=np.int16),
            np.asarray([[1.5, 0.0], [-2.5, 3.5]], dtype=np.float32),
            np.asarray([[1.5, 0.0], [-2.5, 3.5]], dtype=np.float64),
            np.asarray([[1.5, 0.0], [-2.5, 3.5]], dtype=np.longdouble),
        ],
    )
    def test_converts_dense_real_values_and_preserves_dtype(self, data: np.ndarray) -> None:
        before = data.copy()

        result = as_csr_array(data)

        _assert_canonical_csr(result)
        assert result.shape == data.shape
        assert result.dtype == data.dtype
        np.testing.assert_array_equal(result.toarray(), data)
        np.testing.assert_array_equal(data, before)

    @pytest.mark.parametrize(
        "factory",
        [
            sparse.bsr_array,
            sparse.coo_array,
            sparse.csc_array,
            sparse.csr_array,
            sparse.dia_array,
            sparse.dok_array,
            sparse.lil_array,
            sparse.bsr_matrix,
            sparse.coo_matrix,
            sparse.csc_matrix,
            sparse.csr_matrix,
            sparse.dia_matrix,
            sparse.dok_matrix,
            sparse.lil_matrix,
        ],
    )
    def test_converts_every_two_dimensional_scipy_sparse_format(
        self,
        factory: Callable[[np.ndarray], sparse.sparray | sparse.spmatrix],
    ) -> None:
        expected = np.asarray([[0, 2, 0], [3, 0, 4]], dtype=np.int32)
        source = factory(expected)

        result = as_csr_array(source)

        _assert_canonical_csr(result)
        assert result.dtype == expected.dtype
        np.testing.assert_array_equal(result.toarray(), expected)

    def test_reuses_canonical_csr_array(self) -> None:
        source = sparse.csr_array([[1, 0], [0, 2]], dtype=np.int32)

        assert as_csr_array(source) is source

    def test_checks_finite_values_in_bounded_chunks(self, mocker: MockerFixture) -> None:
        size = 1_100_000
        source = sparse.csr_array(
            (
                np.ones(size, dtype=np.float32),
                np.arange(size, dtype=np.int32),
                np.asarray([0, size], dtype=np.int32),
            ),
            shape=(1, size),
        )
        original_isfinite = np.isfinite
        checked_sizes: list[int] = []

        def record_size(values: np.ndarray) -> np.ndarray:
            checked_sizes.append(values.size)
            return original_isfinite(values)

        mocker.patch.object(np, "isfinite", side_effect=record_size)

        assert as_csr_array(source) is source
        assert sum(checked_sizes) == size
        assert max(checked_sizes) < size

    def test_canonical_csr_matrix_can_share_storage(self) -> None:
        source = sparse.csr_matrix([[1, 0], [0, 2]], dtype=np.int32)

        result = as_csr_array(source)

        _assert_canonical_csr(result)
        assert np.shares_memory(result.data, source.data)
        assert np.shares_memory(result.indices, source.indices)
        assert np.shares_memory(result.indptr, source.indptr)

    def test_canonicalizes_without_mutating_noncanonical_csr_input(self) -> None:
        source = sparse.csr_array(
            (
                np.asarray([1, 2, 3], dtype=np.int16),
                np.asarray([1, 0, 0], dtype=np.int32),
                np.asarray([0, 2, 3], dtype=np.int32),
            ),
            shape=(2, 2),
        )
        data_before = source.data.copy()
        indices_before = source.indices.copy()
        indptr_before = source.indptr.copy()
        assert not source.has_canonical_format

        result = as_csr_array(source)

        _assert_canonical_csr(result)
        assert result is not source
        np.testing.assert_array_equal(source.data, data_before)
        np.testing.assert_array_equal(source.indices, indices_before)
        np.testing.assert_array_equal(source.indptr, indptr_before)
        assert not source.has_canonical_format

    def test_sums_coo_duplicates_without_mutating_input(self) -> None:
        source = sparse.coo_array(
            (
                np.asarray([2, 3, 4], dtype=np.int16),
                (
                    np.asarray([0, 0, 1], dtype=np.int32),
                    np.asarray([1, 1, 0], dtype=np.int32),
                ),
            ),
            shape=(2, 2),
        )
        data_before = source.data.copy()
        rows_before = source.row.copy()
        columns_before = source.col.copy()

        result = as_csr_array(source)

        _assert_canonical_csr(result)
        np.testing.assert_array_equal(result.toarray(), np.asarray([[0, 5], [4, 0]], dtype=np.int16))
        np.testing.assert_array_equal(source.data, data_before)
        np.testing.assert_array_equal(source.row, rows_before)
        np.testing.assert_array_equal(source.col, columns_before)
        assert not source.has_canonical_format

    @pytest.mark.parametrize("shape", [(0, 0), (0, 4), (3, 0)])
    def test_preserves_empty_shapes(self, shape: tuple[int, int]) -> None:
        result = as_csr_array(np.empty(shape, dtype=np.float32))

        _assert_canonical_csr(result)
        assert result.shape == shape
        assert result.dtype == np.float32
        assert result.nnz == 0

    @pytest.mark.parametrize(
        ("data", "error", "match"),
        [
            (np.asarray([1, 2]), ValueError, "two-dimensional"),
            (np.ones((1, 1, 1)), ValueError, "two-dimensional"),
            (np.asarray([[1]], dtype=np.float16), TypeError, "Float16"),
            (np.asarray([[1 + 2j]]), TypeError, "complex"),
            (np.asarray([[object()]], dtype=object), TypeError, "object"),
            (np.asarray([["1"]]), TypeError, "string"),
            (np.asarray([[(1, 2)]], dtype=[("a", "i4"), ("b", "i4")]), TypeError, "structured"),
            (np.asarray([[np.nan]]), ValueError, "finite"),
            (np.asarray([[np.inf]]), ValueError, "finite"),
        ],
    )
    def test_rejects_invalid_dense_values(
        self,
        data: np.ndarray,
        error: type[Exception],
        match: str,
    ) -> None:
        with pytest.raises(error, match=match):
            as_csr_array(data)

    @pytest.mark.parametrize(
        ("data", "error", "match"),
        [
            (sparse.csr_array([1, 2]), ValueError, "two-dimensional"),
            (sparse.csc_array([[np.nan]]), ValueError, "finite"),
            (sparse.coo_matrix([[1 + 2j]]), TypeError, "complex"),
        ],
    )
    def test_rejects_invalid_sparse_values(
        self,
        data: sparse.sparray | sparse.spmatrix,
        error: type[Exception],
        match: str,
    ) -> None:
        with pytest.raises(error, match=match):
            as_csr_array(data)

    @pytest.mark.parametrize(
        "data",
        [
            [[1, 0], [0, 2]],
            np.ma.array([[1, 0], [0, 2]], mask=False),
        ],
    )
    def test_rejects_unsupported_containers(self, data: object) -> None:
        with pytest.raises(TypeError, match="plain NumPy ndarray or a SciPy sparse"):
            as_csr_array(cast("np.ndarray | sparse.sparray | sparse.spmatrix", data))

    def test_rejects_backed_array_protocol_without_materializing(self) -> None:
        class BackedArraySentinel:
            shape = (2, 2)
            dtype = np.dtype(np.int32)

            def __array__(self) -> np.ndarray:
                message = "must not materialize"
                raise AssertionError(message)

        with pytest.raises(TypeError, match="plain NumPy ndarray or a SciPy sparse"):
            as_csr_array(cast("np.ndarray | sparse.sparray | sparse.spmatrix", BackedArraySentinel()))

    def test_rejects_numpy_matrix(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            data = np.matrix([[1, 0], [0, 2]])

        with pytest.raises(TypeError, match="plain NumPy ndarray or a SciPy sparse"):
            as_csr_array(cast("np.ndarray | sparse.sparray | sparse.spmatrix", data))

    def test_rejects_lazy_input_without_computing(self) -> None:
        @delayed
        def fail_if_computed() -> np.ndarray:
            message = "must not materialize"
            raise AssertionError(message)

        lazy = da.from_delayed(fail_if_computed(), shape=(2, 2), dtype=np.int32)

        with pytest.raises(TypeError, match="plain NumPy ndarray or a SciPy sparse"):
            as_csr_array(cast("np.ndarray | sparse.sparray | sparse.spmatrix", lazy))

    def test_does_not_call_sparse_densification(self, mocker: MockerFixture) -> None:
        source = sparse.csc_matrix([[0, 2], [3, 0]], dtype=np.int32)
        toarray = mocker.patch.object(sparse.csc_matrix, "toarray", side_effect=AssertionError("densified"))
        todense = mocker.patch.object(sparse.csc_matrix, "todense", side_effect=AssertionError("densified"))

        result = as_csr_array(source)

        _assert_canonical_csr(result)
        toarray.assert_not_called()
        todense.assert_not_called()


class TestCsrArrayFromTriplets:
    """Validate coordinate construction, bounds, and duplicate semantics."""

    def test_sums_duplicates_sorts_indices_and_preserves_dtype(self) -> None:
        values = np.asarray([2, 3, 4, -1], dtype=np.int16)
        rows = np.asarray([0, 0, 1, 1], dtype=np.int64)
        columns = np.asarray([1, 1, 0, 0], dtype=np.int64)
        values_before = values.copy()
        rows_before = rows.copy()
        columns_before = columns.copy()

        result = csr_array_from_triplets(values, rows, columns, shape=(2, 2))

        _assert_canonical_csr(result)
        assert result.dtype == np.int16
        assert result.indices.dtype == np.int64
        assert result.indptr.dtype == np.int64
        np.testing.assert_array_equal(result.toarray(), np.asarray([[0, 5], [3, 0]], dtype=np.int16))
        np.testing.assert_array_equal(values, values_before)
        np.testing.assert_array_equal(rows, rows_before)
        np.testing.assert_array_equal(columns, columns_before)

    def test_preserves_explicit_zero_from_duplicate_cancellation(self) -> None:
        result = csr_array_from_triplets(
            np.asarray([2, -2], dtype=np.int8),
            np.asarray([0, 0], dtype=np.int32),
            np.asarray([1, 1], dtype=np.int32),
            shape=(1, 2),
        )

        _assert_canonical_csr(result)
        assert result.nnz == 1
        assert result.data[0] == 0

    @pytest.mark.parametrize("shape", [(0, 0), (0, 5), (4, 0)])
    def test_supports_empty_dimensions(self, shape: tuple[int, int]) -> None:
        result = csr_array_from_triplets(
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.int32),
            shape=shape,
        )

        _assert_canonical_csr(result)
        assert result.shape == shape
        assert result.dtype == np.float32
        assert result.indices.dtype == np.int32
        assert result.indptr.dtype == np.int32
        assert result.nnz == 0

    @pytest.mark.parametrize(
        ("triplets", "shape", "error", "match"),
        [
            ((np.asarray([[1]]), np.asarray([0]), np.asarray([0])), (1, 1), ValueError, "one-dimensional"),
            ((np.asarray([1]), np.asarray([[0]]), np.asarray([0])), (1, 1), ValueError, "one-dimensional"),
            ((np.asarray([1]), np.asarray([0]), np.asarray([[0]])), (1, 1), ValueError, "one-dimensional"),
            ((np.asarray([1, 2]), np.asarray([0]), np.asarray([0])), (1, 1), ValueError, "equal lengths"),
            ((np.asarray([1 + 0j]), np.asarray([0]), np.asarray([0])), (1, 1), TypeError, "complex"),
            (
                (np.asarray([1], dtype=np.float16), np.asarray([0]), np.asarray([0])),
                (1, 1),
                TypeError,
                "Float16",
            ),
            ((np.asarray([np.nan]), np.asarray([0]), np.asarray([0])), (1, 1), ValueError, "finite"),
            ((np.asarray([1]), np.asarray([0.0]), np.asarray([0])), (1, 1), TypeError, "int32 or int64"),
            ((np.asarray([1]), np.asarray([False]), np.asarray([0])), (1, 1), TypeError, "int32 or int64"),
            ((np.asarray([1]), np.asarray([0]), np.asarray([0.0])), (1, 1), TypeError, "int32 or int64"),
            (
                (np.asarray([1]), np.asarray([0], dtype=np.uint32), np.asarray([0], dtype=np.uint32)),
                (1, 1),
                TypeError,
                "int32 or int64",
            ),
            (
                (np.asarray([1]), np.asarray([0], dtype=np.int16), np.asarray([0], dtype=np.int16)),
                (1, 1),
                TypeError,
                "int32 or int64",
            ),
            (
                (np.asarray([1]), np.asarray([0], dtype=np.int32), np.asarray([0], dtype=np.int64)),
                (1, 1),
                TypeError,
                "same dtype",
            ),
            ((np.asarray([1]), np.asarray([-1]), np.asarray([0])), (1, 1), ValueError, "row_indices"),
            ((np.asarray([1]), np.asarray([1]), np.asarray([0])), (1, 1), ValueError, "row_indices"),
            ((np.asarray([1]), np.asarray([0]), np.asarray([-1])), (1, 1), ValueError, "column_indices"),
            ((np.asarray([1]), np.asarray([0]), np.asarray([1])), (1, 1), ValueError, "column_indices"),
            (
                (
                    np.asarray([], dtype=int),
                    np.asarray([], dtype=int),
                    np.asarray([], dtype=int),
                ),
                (-1, 1),
                ValueError,
                "nonnegative",
            ),
        ],
    )
    def test_rejects_invalid_triplets(
        self,
        triplets: tuple[np.ndarray, np.ndarray, np.ndarray],
        shape: tuple[int, int],
        error: type[Exception],
        match: str,
    ) -> None:
        values, rows, columns = triplets
        with pytest.raises(error, match=match):
            csr_array_from_triplets(values, rows, columns, shape=shape)

    @pytest.mark.parametrize("shape", [[1, 1], (1,), (1, 2, 3), (True, 1), (1.5, 1)])
    def test_rejects_invalid_shapes(self, shape: object) -> None:
        empty = np.asarray([], dtype=np.float32)
        indices = np.asarray([], dtype=np.int32)

        with pytest.raises(TypeError, match="pair of nonnegative integers"):
            csr_array_from_triplets(empty, indices, indices, shape=cast("tuple[int, int]", shape))

    def test_accepts_numpy_integer_shape_dimensions(self) -> None:
        result = csr_array_from_triplets(
            np.asarray([1], dtype=np.uint8),
            np.asarray([0], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            shape=cast("tuple[int, int]", (np.int32(1), np.int64(2))),
        )

        _assert_canonical_csr(result)
        assert result.shape == (1, 2)
        assert result.indices.dtype == np.int64
        assert result.indptr.dtype == np.int64

    def test_rejects_shape_that_requires_wider_indices(self) -> None:
        empty = np.asarray([], dtype=np.float32)
        indices = np.asarray([], dtype=np.int32)
        shape = (0, np.iinfo(np.int32).max + 1)

        with pytest.raises(ValueError, match="cannot be represented by int32"):
            csr_array_from_triplets(empty, indices, indices, shape=shape)

    @pytest.mark.parametrize("parameter", ["values", "row_indices", "column_indices"])
    def test_rejects_non_ndarray_triplet_inputs(self, parameter: str) -> None:
        arrays: dict[str, object] = {
            "values": np.asarray([1]),
            "row_indices": np.asarray([0]),
            "column_indices": np.asarray([0]),
        }
        arrays[parameter] = [0]

        with pytest.raises(TypeError, match=f"{parameter} must be a plain NumPy ndarray"):
            csr_array_from_triplets(
                cast("np.ndarray", arrays["values"]),
                cast("np.ndarray", arrays["row_indices"]),
                cast("np.ndarray", arrays["column_indices"]),
                shape=(1, 1),
            )

    def test_rejects_nonfinite_duplicate_sum(self) -> None:
        maximum = np.finfo(np.float32).max
        with (
            np.errstate(over="ignore"),
            pytest.raises(ValueError, match="summed values must contain only finite"),
        ):
            csr_array_from_triplets(
                np.asarray([maximum, maximum], dtype=np.float32),
                np.asarray([0, 0], dtype=np.int32),
                np.asarray([0, 0], dtype=np.int32),
                shape=(1, 1),
            )
