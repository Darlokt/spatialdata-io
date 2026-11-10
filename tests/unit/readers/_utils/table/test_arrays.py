"""Tests for canonical sparse-array construction."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import dask.array as da
import numpy as np
import pytest
from dask import delayed
from scipy import sparse

from spatialdata_io.readers._utils.table import as_csr_array
from spatialdata_io.readers._utils.table._arrays import validate_compressed_sparse

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

    @pytest.mark.parametrize("factory", [sparse.csr_array, sparse.csc_array, sparse.csr_matrix, sparse.csc_matrix])
    @pytest.mark.parametrize("flags", [True, False])
    def test_rejects_bad_indices_before_canonicalization_or_conversion(
        self,
        factory: Callable[[np.ndarray], sparse.csr_array | sparse.csc_array | sparse.csr_matrix | sparse.csc_matrix],
        *,
        flags: bool,
        mocker: MockerFixture,
    ) -> None:
        source = factory(np.eye(2, dtype=np.int16))
        source.has_canonical_format = flags
        source.has_sorted_indices = flags
        source.indices[0] = 2
        mocker.patch.object(source, "copy", side_effect=AssertionError("copy before validation"))
        mocker.patch.object(source, "tocsr", side_effect=AssertionError("conversion before validation"))
        mocker.patch.object(source, "sum_duplicates", side_effect=AssertionError("summation before validation"))

        with pytest.raises(ValueError, match="indices must be < 2"):
            as_csr_array(source)

        assert source.indices[0] == 2
        assert source.__dict__["_has_canonical_format"] is flags

    @pytest.mark.parametrize("factory", [sparse.csr_array, sparse.csc_array, sparse.csr_matrix, sparse.csc_matrix])
    @pytest.mark.parametrize("index_dtype", ["int8", "int16", "uint8", "uint16", "uint32", ">i4", ">i8", ">u4"])
    @pytest.mark.parametrize("canonical", [True, False])
    def test_converts_supported_index_storage_without_changing_source_buffers(
        self,
        factory: Callable[[np.ndarray], sparse.csr_array | sparse.csc_array | sparse.csr_matrix | sparse.csc_matrix],
        index_dtype: str,
        *,
        canonical: bool,
    ) -> None:
        source = factory(np.zeros((2, 2), dtype=np.int16))
        source.data = np.asarray([1, 2, 3], dtype=np.int16)
        source.indices = np.asarray([0, 1, 0] if canonical else [1, 0, 0], dtype=index_dtype)
        source.indptr = np.asarray([0, 2, 3] if canonical else [0, 3, 3], dtype=index_dtype)
        before = [(buffer, buffer.copy()) for buffer in (source.data, source.indices, source.indptr)]

        result = as_csr_array(source)

        expected = np.asarray([[1, 2], [3, 0]] if canonical else [[5, 1], [0, 0]], dtype=np.int16)
        if isinstance(source, (sparse.csc_array, sparse.csc_matrix)):
            expected = expected.T
        np.testing.assert_array_equal(result.toarray(), expected)
        _assert_canonical_csr(result)
        for buffer, values in before:
            np.testing.assert_array_equal(buffer, values)
        assert source.indices is before[1][0]
        assert source.indptr is before[2][0]


class TestValidateCompressedSparse:
    """Check borrowed compressed structure through SciPy without changing the source."""

    @pytest.mark.parametrize("factory", [sparse.csr_array, sparse.csc_array, sparse.csr_matrix, sparse.csc_matrix])
    @pytest.mark.parametrize("dtype", [np.int16, np.float32])
    def test_accepts_borrowed_noncanonical_buffers_without_changes(
        self,
        factory: Callable[[np.ndarray], sparse.csr_array | sparse.csc_array | sparse.csr_matrix | sparse.csc_matrix],
        dtype: type[np.int16 | np.float32],
        mocker: MockerFixture,
    ) -> None:
        source = factory(np.zeros((3, 3), dtype=dtype))
        source.data = np.asarray([1, 2, 0, 4], dtype=dtype)
        source.indices = np.asarray([2, 0, 0, 1], dtype=np.int32)
        source.indptr = np.asarray([0, 3, 3, 4], dtype=np.int32)
        before = source.__dict__.copy()
        buffers = [(buffer, buffer.copy()) for buffer in (source.data, source.indices, source.indptr)]
        mocker.patch.object(source, "check_format", side_effect=AssertionError("mutating format check"))
        mocker.patch.object(source, "copy", side_effect=AssertionError("sparse copy"))

        assert validate_compressed_sparse(source) is None

        for name, value in before.items():
            assert source.__dict__[name] is value
        for original, expected in buffers:
            np.testing.assert_array_equal(original, expected)

    @pytest.mark.parametrize("factory", [sparse.csr_array, sparse.csc_array, sparse.csr_matrix, sparse.csc_matrix])
    @pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0), (2, 3)])
    def test_accepts_empty_dimensions_and_spans(
        self,
        factory: Callable[[np.ndarray], sparse.csr_array | sparse.csc_array | sparse.csr_matrix | sparse.csc_matrix],
        shape: tuple[int, int],
    ) -> None:
        validate_compressed_sparse(factory(np.zeros(shape, dtype=np.int32)))

    @pytest.mark.parametrize("dtype", ["int32", "int64", "uint32", ">i4", ">i8"])
    def test_accepts_integer_index_storage_without_casting(self, dtype: str) -> None:
        source = sparse.csr_array([[1, 0]], dtype=np.int16)
        source.indices = source.indices.astype(dtype)
        source.indptr = source.indptr.astype(dtype)
        before = (source.indices, source.indptr)

        validate_compressed_sparse(source)

        assert source.indices is before[0]
        assert source.indptr is before[1]

    def test_accepts_unused_capacity_without_inspecting_inactive_indices(self) -> None:
        source = sparse.csr_array([[1, 0]], dtype=np.int16)
        source.data = np.asarray([1, 2, 3], dtype=np.int16)
        source.indices = np.asarray([0, -1, 100], dtype=np.int32)
        before = (source.data, source.indices, source.indptr)

        validate_compressed_sparse(source)

        assert source.data is before[0]
        assert source.indices is before[1]
        assert source.indptr is before[2]
        assert source.data.size == 3

    @pytest.mark.parametrize(
        ("name", "buffer", "match"),
        [
            ("data", np.ones((1, 1)), "should be 1-D"),
            ("indices", np.asarray([0, 1], dtype=np.int32), "same size"),
            ("indptr", np.asarray([0, 1, 1], dtype=np.int32), "index pointer size 3 should be 2"),
            ("indptr", np.asarray([1, 1], dtype=np.int32), "should start with 0"),
            ("indptr", np.asarray([0, 2], dtype=np.int32), "Last value of index pointer"),
            ("indptr", np.asarray([0, -1], dtype=np.int32), "nondecreasing from zero"),
            ("indices", np.asarray([-1], dtype=np.int32), "indices must be >= 0"),
            ("indices", np.asarray([2], dtype=np.int32), "indices must be < 2"),
        ],
    )
    def test_rejects_each_malformation_family_without_changing_source(
        self, name: str, buffer: np.ndarray, match: str
    ) -> None:
        source = sparse.csr_array([[1, 0]], dtype=np.int16)
        source.has_canonical_format = True
        source.has_sorted_indices = True
        setattr(source, name, buffer)

        with pytest.raises(ValueError, match=match):
            validate_compressed_sparse(source)

        assert getattr(source, name) is buffer
        assert source.__dict__["_has_canonical_format"] is True
        assert source.__dict__["_has_sorted_indices"] is True

    @pytest.mark.parametrize("indptr", [[0, 1, 0], [0, -1, 1], [0, 2**63 - 1, 1]])
    def test_rejects_decreasing_pointers(self, indptr: list[int]) -> None:
        source = sparse.csr_array([[1], [0]], dtype=np.int16)
        source.__dict__["indptr"] = np.asarray(indptr, dtype=np.int64)

        with pytest.raises(ValueError, match=r"non-?decreasing"):
            validate_compressed_sparse(source)

        np.testing.assert_array_equal(source.indptr, indptr)

    def test_rejects_one_dimensional_sparse_input(self) -> None:
        with pytest.raises(ValueError, match="two-dimensional"):
            validate_compressed_sparse(sparse.csr_array([1, 0]))

    @pytest.mark.parametrize("factory", [sparse.csr_array, sparse.csc_array, sparse.csr_matrix, sparse.csc_matrix])
    def test_checks_minor_dimension_for_rectangular_storage(
        self,
        factory: Callable[[np.ndarray], sparse.csr_array | sparse.csc_array | sparse.csr_matrix | sparse.csc_matrix],
    ) -> None:
        source = factory(np.asarray([[0, 0, 1], [0, 1, 0]], dtype=np.int16))
        validate_compressed_sparse(source)
        bound = 3 if isinstance(source, (sparse.csr_array, sparse.csr_matrix)) else 2
        source.indices[0] = bound
        with pytest.raises(ValueError, match=f"indices must be < {bound}"):
            validate_compressed_sparse(source)

    def test_accepts_all_unused_capacity_with_zero_terminal_pointer(self) -> None:
        source = sparse.csr_array((1, 0), dtype=np.int16)
        source.data = np.asarray([4], dtype=np.int16)
        source.indices = np.asarray([-1], dtype=np.int32)

        validate_compressed_sparse(source)

        assert source.data.size == 1
        np.testing.assert_array_equal(source.indptr, [0, 0])
