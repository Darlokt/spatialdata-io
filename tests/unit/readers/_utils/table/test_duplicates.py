"""Imported duplicate coordinates preserve values through exact promotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import sparse

from spatialdata_io.readers._utils.table import SparseDuplicateSummary
from spatialdata_io.readers._utils.table._duplicates import _exact_integer_cast, reduce_duplicates

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytest_mock import MockerFixture


def _source(values: Sequence[int | float], dtype: str, storage: str) -> sparse.csr_array | sparse.csc_array:
    buffers = (np.asarray(values, dtype=dtype), np.zeros(len(values), dtype=np.int32), np.array([0, len(values)]))
    return sparse.csr_array(buffers, shape=(1, 2)) if storage == "csr" else sparse.csc_array(buffers, shape=(2, 1))


class TestReduceDuplicates:
    @pytest.mark.parametrize("storage", ["csr", "csc"])
    @pytest.mark.parametrize(
        ("dtype", "values", "expected_dtype"),
        [
            ("int8", [100, 100], "float32"),
            ("int8", [-100, -100], "float32"),
            ("uint8", [150, 150], "float32"),
            ("int16", [30000, 30000], "float32"),
            ("uint16", [65535, 1], "float32"),
            ("int16", [2**15 - 1, 2**15 - 1], "float32"),
            ("int32", [2**31 - 1, 1], "float32"),
            ("int32", [2**31 - 1, 2], "float64"),
            ("int32", [-(2**31), -(2**31)], "float32"),
            ("uint32", [2**32 - 1, 1], "float32"),
            ("uint32", [2**32 - 1, 2], "float64"),
            ("int8", [100, 100, -100], "int8"),
            ("int32", [2**30, 2**30, -(2**30)], "int32"),
            ("int16", [100, 100], "int16"),
            ("int8", [100, -100], "int8"),
        ],
    )
    def test_exact_sums_and_required_promotion(
        self, storage: str, dtype: str, values: list[int], expected_dtype: str
    ) -> None:
        source = _source(values, dtype, storage)
        before = source.data.copy(), source.indices.copy(), source.indptr.copy()
        result = reduce_duplicates(source)
        assert result.dtype == np.dtype(expected_dtype)
        assert result.has_canonical_format
        assert result.nnz == 1
        assert int(result.data[0]) == sum(values)
        assert isinstance(result, sparse.csr_array)
        for actual, original in zip((source.data, source.indices, source.indptr), before, strict=True):
            np.testing.assert_array_equal(actual, original)

    @pytest.mark.parametrize(
        ("dtype", "values"),
        [("int64", [2**62, 2**62]), ("int64", [-(2**63), -1]), ("uint64", [2**64 - 1, 1])],
    )
    def test_partial_sums_beyond_64_bits_raise(self, dtype: str, values: list[int]) -> None:
        source = _source(values, dtype, "csr")
        with pytest.raises(ValueError, match=f"exceed {dtype} arithmetic"):
            reduce_duplicates(source)
        np.testing.assert_array_equal(source.data, np.array(values, dtype=dtype))

    def test_promoted_completed_sum_must_be_exact_in_float64(self) -> None:
        # The odd uint64 total exceeds 2**53 but fits the 64-bit reduction.
        # About 16 MiB of source buffers suffices to cross this boundary with
        # uint32 contributions; no malformed storage or private state is used.
        entries = 2**21 + 1
        source = sparse.csr_array(
            (np.full(entries, 2**32 - 1, dtype=np.uint32), np.zeros(entries, dtype=np.int32), np.array([0, entries])),
            shape=(1, 1),
        )
        with pytest.raises(ValueError, match="no supported lossless"):
            reduce_duplicates(source)
        assert source.nnz == entries
        assert source.data[0] == 2**32 - 1

    @pytest.mark.parametrize("storage", ["csr", "csc"])
    def test_unsorted_duplicates_are_widened_before_addition(self, storage: str) -> None:
        buffers = (np.array([2**31 - 1, 1, 3], dtype=np.int32), np.array([1, 1, 0]), np.array([0, 3]))
        source = (
            sparse.csr_array(buffers, shape=(1, 2)) if storage == "csr" else sparse.csc_array(buffers, shape=(2, 1))
        )

        result = reduce_duplicates(source)

        assert result.dtype == np.dtype(np.float32)
        assert result.has_canonical_format
        np.testing.assert_array_equal(result.data, [3, 2**31])
        np.testing.assert_array_equal(source.data, [2**31 - 1, 1, 3])
        np.testing.assert_array_equal(source.indices, [1, 1, 0])

    @pytest.mark.parametrize("storage", ["csr", "csc"])
    @pytest.mark.parametrize(("unaffected", "dtype"), [(7, "float32"), (2**24 + 1, "float64")])
    def test_promoted_dtype_includes_unaffected_values(self, storage: str, unaffected: int, dtype: str) -> None:
        buffers = (np.array([2**31 - 1, 1, unaffected], dtype=np.int32), np.array([0, 0, 1]), np.array([0, 3]))
        source = (
            sparse.csr_array(buffers, shape=(1, 2)) if storage == "csr" else sparse.csc_array(buffers, shape=(2, 1))
        )
        result = reduce_duplicates(source)
        assert result.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(result.data, [2**31, unaffected])

    def test_unsorted_unique_entries_are_not_summed(self) -> None:
        source = sparse.csr_array(
            (np.array([127, 126], dtype=np.int8), np.array([1, 0]), np.array([0, 2])), shape=(1, 2)
        )
        result = reduce_duplicates(source)
        assert result.dtype == source.dtype
        np.testing.assert_array_equal(result.data, [126, 127])
        assert result.nnz == source.nnz

    @pytest.mark.parametrize("storage", ["csr", "csc"])
    @pytest.mark.parametrize("dtype", ["int64", "uint64"])
    def test_unsorted_unique_integer_extrema_need_no_sum_bound(self, storage: str, dtype: str) -> None:
        maximum = np.iinfo(dtype).max
        buffers = (np.array([maximum, 1], dtype=dtype), np.array([1, 0]), np.array([0, 2]))
        source = (
            sparse.csr_array(buffers, shape=(1, 2)) if storage == "csr" else sparse.csc_array(buffers, shape=(2, 1))
        )

        result = reduce_duplicates(source)

        assert result.dtype == source.dtype
        assert result.has_canonical_format
        np.testing.assert_array_equal(result.data, np.array([1, maximum], dtype=dtype))
        np.testing.assert_array_equal(source.data, np.array([maximum, 1], dtype=dtype))

    def test_many_small_compressed_spans_retain_native_dtype(self) -> None:
        source = sparse.csr_array(
            (np.tile(np.array([20, 30], dtype=np.int8), 100), np.zeros(200, dtype=np.int32), np.arange(0, 201, 2)),
            shape=(100, 1),
        )
        result = reduce_duplicates(source)
        assert result.dtype == np.dtype(np.int8)
        assert result.has_canonical_format
        np.testing.assert_array_equal(result.data, np.full(100, 50, dtype=np.int8))

    def test_late_compressed_spans_are_included_in_the_safety_bound(self, mocker: MockerFixture) -> None:
        mocker.patch("spatialdata_io.readers._utils.table._duplicates._BOUND_BATCH_SIZE", 2)
        source = sparse.csr_array(
            (np.array([1, 1, 150, 150], dtype=np.uint8), np.zeros(4, dtype=np.int32), np.array([0, 1, 2, 4])),
            shape=(3, 1),
        )
        result = reduce_duplicates(source)
        assert result.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(result.data, [1, 1, 300])

    @pytest.mark.parametrize("storage", ["csr", "csc"])
    def test_float_and_boolean_semantics_remain(self, storage: str) -> None:
        result = reduce_duplicates(_source([0.5, -0.25], "float32", storage))
        assert result.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(result.data, [0.25])
        result = reduce_duplicates(_source([1, 1], "bool", storage))
        assert result.dtype == np.dtype(bool)
        np.testing.assert_array_equal(result.data, [True])


class TestExactIntegerCast:
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("source_dtype", ["int64", "uint64"])
    def test_exact_powers_above_consecutive_integer_precision(self, dtype: str, source_dtype: str) -> None:
        values = np.array([0, 2**54], dtype=source_dtype)
        assert _exact_integer_cast(values, np.dtype(dtype))

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("source_dtype", ["int64", "uint64"])
    def test_rounded_upper_limit_is_not_cast_back(self, dtype: str, source_dtype: str) -> None:
        values = np.array([np.iinfo(source_dtype).max], dtype=source_dtype)
        assert not _exact_integer_cast(values, np.dtype(dtype))

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_signed_minimum_is_exact(self, dtype: str) -> None:
        assert _exact_integer_cast(np.array([-(2**63)], dtype=np.int64), np.dtype(dtype))

    def test_inexact_values_in_later_batches(self, mocker: MockerFixture) -> None:
        mocker.patch("spatialdata_io.readers._utils.table._duplicates._BOUND_BATCH_SIZE", 2)
        assert not _exact_integer_cast(np.array([1, 2, 2**53 + 1], dtype=np.int64), np.dtype(np.float64))


class TestSparseDuplicateSummary:
    def test_summary_reports_promotion_and_safe_sums(self) -> None:
        promoted = SparseDuplicateSummary("X", 1, np.dtype(np.int8), np.dtype(np.float32))
        assert (
            str(promoted) == "combined 1 duplicate sparse entry in X; promoted int8 to float32 to preserve exact sums"
        )
        safe = SparseDuplicateSummary("layers['counts']", 2, np.dtype(np.int16), np.dtype(np.int16))
        assert str(safe) == "combined 2 duplicate sparse entries in layers['counts']"
