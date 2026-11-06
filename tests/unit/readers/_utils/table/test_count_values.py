"""Exact sparse count storage and family-wide dtype selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.table import _count_values, count_accumulation_dtype, finalize_count_arrays

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

_CONTEXT = ReaderErrorContext("test_reader", "completed counts")


class TestCountAccumulationDtype:
    @pytest.mark.parametrize(
        ("bound", "expected"),
        [
            (0, "float32"),
            (1, "float32"),
            (2**24 - 1, "float32"),
            (2**24, "float32"),
            (2**24 + 1, "float64"),
            (2**24 + 2, "float64"),
            (2**53 - 1, "float64"),
            (2**53, "float64"),
        ],
    )
    def test_selects_dtype_protecting_every_partial_sum(self, bound: int, expected: str) -> None:
        assert count_accumulation_dtype(bound) == np.dtype(expected)

    @pytest.mark.parametrize("bound", [2**53 + 1, 2**53 + 2, 2**64, 10**400])
    def test_inconclusive_bound_leaves_fallback_to_reader(self, bound: int) -> None:
        assert count_accumulation_dtype(bound) is None

    @pytest.mark.parametrize("bound", [-1, -(10**400)])
    def test_rejects_negative_bounds(self, bound: int) -> None:
        with pytest.raises(ValueError, match="nonnegative"):
            count_accumulation_dtype(bound)

    @pytest.mark.parametrize("bound", [True, False, 1.0, 0.5, np.nan, np.inf, "1", None, np.int64(1)])
    def test_rejects_non_python_integers(self, bound: object) -> None:
        with pytest.raises(TypeError, match="Python integer"):
            count_accumulation_dtype(cast("int", bound))


class TestFinalizeCountArrays:
    @pytest.mark.parametrize("value", [-1, 2**53 + 1])
    def test_numerical_errors_identify_source(self, tmp_path: Path, value: int) -> None:
        path = tmp_path / "counts.h5"
        source = sparse.csr_array([[value]], dtype=np.int64)
        with pytest.raises(ReaderFormatError, match=str(path)) as caught:
            finalize_count_arrays((source,), context=_CONTEXT, path=path)
        assert caught.value.path == path
        assert caught.value.context == _CONTEXT

    @pytest.mark.parametrize(
        ("dtype", "value", "expected"),
        [
            ("bool", 1, "float32"),
            ("int8", 127, "float32"),
            ("uint16", 65535, "float32"),
            ("int32", 2**24, "float32"),
            ("int64", 2**24 + 1, "float64"),
            ("int64", 2**24 + 2, "float32"),
            ("int64", 2**53 + 2, "float64"),
            ("uint64", 2**63, "float32"),
            ("uint64", 2**64 - 2048, "float64"),
            ("float32", 42, "float32"),
            ("float64", 2**24 + 1, "float64"),
        ],
    )
    def test_selects_by_actual_values(self, dtype: str, value: int, expected: str) -> None:
        source = sparse.csr_array(np.array([[value]], dtype=dtype))
        (result,) = finalize_count_arrays((source,), context=_CONTEXT)
        assert result.dtype == np.dtype(expected)
        assert int(result.data[0]) == value
        assert result.has_canonical_format
        assert isinstance(result, sparse.sparray)
        assert not isinstance(result, sparse.spmatrix)

    @pytest.mark.parametrize(("dtype", "value"), [("int64", 2**53 + 1), ("int64", 2**63 - 1), ("uint64", 2**64 - 1)])
    def test_rejects_inexact_integer_float64(self, dtype: str, value: int) -> None:
        source = sparse.csr_array(np.array([[value]], dtype=dtype))
        with pytest.raises(ReaderFormatError, match="not exactly representable"):
            finalize_count_arrays((source,), context=_CONTEXT)
        assert int(source.data[0]) == value

    @pytest.mark.parametrize("value", [-1, 0.5, np.nan, np.inf, -np.inf])
    def test_rejects_invalid_counts(self, value: float) -> None:
        source = sparse.csr_array([[value]])
        with pytest.raises(ReaderFormatError, match="finite nonnegative integer counts"):
            finalize_count_arrays((source,), context=_CONTEXT)

    @pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "float32", "float64"])
    def test_negative_storage_values_are_rejected_before_conversion(self, dtype: str) -> None:
        source = sparse.csr_array(np.array([[-1, 0, 1]], dtype=dtype))
        with pytest.raises(ReaderFormatError, match="nonnegative"):
            finalize_count_arrays((source,), context=_CONTEXT)
        np.testing.assert_array_equal(source.data, [-1, 1])
        assert source.dtype == np.dtype(dtype)

    def test_family_dtype_includes_empty_arrays_and_ignores_order(self) -> None:
        ordinary = sparse.csr_array([[1]], dtype=np.float32)
        wide = sparse.csr_array([[2**24 + 1]], dtype=np.int32)
        empty = sparse.csr_array((1, 0), dtype=np.uint64)
        for family in ((ordinary, wide, empty), (empty, wide, ordinary)):
            results = finalize_count_arrays(family, context=_CONTEXT)
            assert all(result.dtype == np.dtype(np.float64) for result in results)
            assert [result.shape for result in results] == [array.shape for array in family]

    @pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
    def test_preserves_structure_zeros_and_ownership(self, index_dtype: type[np.int32 | np.int64]) -> None:
        source = sparse.csr_array(
            (
                np.array([0, 8], dtype=np.int32),
                np.array([0, 2], dtype=index_dtype),
                np.array([0, 2], dtype=index_dtype),
            ),
            shape=(1, 4),
        )
        (result,) = finalize_count_arrays((source,), context=_CONTEXT)
        assert result.nnz == 2
        np.testing.assert_array_equal(result.data, [0, 8])
        assert result.indices.dtype == source.indices.dtype
        assert result.indptr.dtype == source.indptr.dtype
        assert np.shares_memory(result.indices, source.indices)
        assert np.shares_memory(result.indptr, source.indptr)
        assert not np.shares_memory(result.data, source.data)
        result.data[1] = 9
        assert source.data[1] == 8

    def test_reuses_suitable_array_without_densification(self, mocker: MockerFixture) -> None:
        source = sparse.csr_array([[1, 0, 2]], dtype=np.float32)
        mocker.patch.object(source, "toarray", side_effect=AssertionError("densification"))
        mocker.patch.object(source, "astype", side_effect=AssertionError("unnecessary cast"))
        assert finalize_count_arrays((source,), context=_CONTEXT)[0] is source

    def test_validates_late_chunks(self, mocker: MockerFixture) -> None:
        mocker.patch("spatialdata_io.readers._utils.table._count_values._VALIDATION_BATCH_SIZE", 2)
        source = sparse.csr_array([[1, 2, -3]], dtype=np.int32)
        with pytest.raises(ReaderFormatError, match="negative"):
            finalize_count_arrays((source,), context=_CONTEXT)

    def test_empty_family(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            finalize_count_arrays((), context=_CONTEXT)

    @pytest.mark.parametrize("dtype", ["int64", "uint64", "float64", "longdouble"])
    def test_skips_elementwise_casts_for_values_within_exact_integer_range(
        self, dtype: str, mocker: MockerFixture
    ) -> None:
        cast_check = mocker.spy(_count_values, "_exact_float_cast")
        source = sparse.csr_array(np.array([[0, 1, 2**24]], dtype=dtype))

        (result,) = finalize_count_arrays((source,), context=_CONTEXT)

        cast_check.assert_not_called()
        assert result.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(result.data, [1, 2**24])

    def test_checks_only_float32_between_exact_integer_ranges(self, mocker: MockerFixture) -> None:
        cast_check = mocker.spy(_count_values, "_exact_float_cast")
        exact = sparse.csr_array(np.array([[2**25]], dtype=np.int64))
        inexact = sparse.csr_array(np.array([[2**24 + 1]], dtype=np.int64))

        assert finalize_count_arrays((exact,), context=_CONTEXT)[0].dtype == np.dtype(np.float32)
        assert finalize_count_arrays((inexact,), context=_CONTEXT)[0].dtype == np.dtype(np.float64)

        assert [call.args[1] for call in cast_check.call_args_list] == [np.dtype(np.float32)] * 2

    def test_noncanonical_and_unsupported_values(self) -> None:
        source = sparse.csr_array((np.array([1, 2]), np.array([0, 0]), np.array([0, 2])), shape=(1, 1))
        with pytest.raises(ValueError, match="Complete duplicate aggregation"):
            finalize_count_arrays((source,), context=_CONTEXT)
        with pytest.raises(TypeError, match="real floating"):
            finalize_count_arrays((sparse.csr_array([[1j]]),), context=_CONTEXT)

    @pytest.mark.parametrize("shape", [(0, 4), (3, 0), (2, 3)])
    def test_zero_empty_counts(self, shape: tuple[int, int]) -> None:
        (result,) = finalize_count_arrays((sparse.csr_array(shape, dtype=np.int64),), context=_CONTEXT)
        assert result.shape == shape
        assert result.nnz == 0
        assert result.dtype == np.dtype(np.float32)

    def test_real_float_overflow_and_noninteger_longdouble(self) -> None:
        source = sparse.csr_array([[np.finfo(np.longdouble).max]], dtype=np.longdouble)
        with pytest.raises(ReaderFormatError, match="not exactly representable"):
            finalize_count_arrays((source,), context=_CONTEXT)
