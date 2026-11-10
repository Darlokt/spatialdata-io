"""Lossless normalization of imported compressed sparse duplicate entries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from spatialdata_io.readers._utils.table._arrays import (
    _compressed_to_csr,
    _validate_value_dtype,
    validate_compressed_sparse,
)

_BOUND_BATCH_SIZE = 1_048_576


@dataclass(frozen=True, slots=True)
class SparseDuplicateSummary:
    """Describe combined storage entries for a reader-owned warning summary.

    Parameters
    ----------
    slot
        Reader-visible expression slot, optionally qualified by table/artifact.
    combined_entries
        Number of redundant stored entries combined into existing coordinates.
    source_dtype
        Original stored-value dtype.
    result_dtype
        Canonical result dtype, promoted only when required for exact sums.
    """

    slot: str
    combined_entries: int
    source_dtype: np.dtype[np.generic]
    result_dtype: np.dtype[np.generic]

    def __str__(self) -> str:
        """Describe normalization without exposing expression values or IDs."""
        noun = "entry" if self.combined_entries == 1 else "entries"
        message = f"combined {self.combined_entries} duplicate sparse {noun} in {self.slot}"
        if self.source_dtype != self.result_dtype:
            message += f"; promoted {self.source_dtype} to {self.result_dtype} to preserve exact sums"
        return message


def _maximum_span(indptr: np.ndarray[tuple[int], np.dtype[np.integer]]) -> int:
    """Return the most stored entries of one compressed row or column.

    Differences are taken in bounded batches so the scan never allocates a
    second pointer-sized array.
    """
    maximum = 0
    for start in range(0, len(indptr) - 1, _BOUND_BATCH_SIZE):
        spans = np.diff(indptr[start : start + _BOUND_BATCH_SIZE + 1])
        maximum = max(maximum, int(spans.max(initial=0)))
    return maximum


def _exact_integer_cast(
    values: np.ndarray[tuple[int], np.dtype[np.integer]], dtype: np.dtype[np.float32 | np.float64]
) -> bool:
    """Check floating representability without rounding integers during comparison."""
    limits = np.iinfo(values.dtype)
    for start in range(0, values.size, _BOUND_BATCH_SIZE):
        batch = values[start : start + _BOUND_BATCH_SIZE]
        converted = batch.astype(dtype)
        # The exclusive upper bound is a power of two, exactly representable
        # in both float widths. Casting that bound back to integers overflows.
        if np.any(converted >= limits.max + 1) or np.any(converted < limits.min):
            return False
        if not np.array_equal(converted.astype(values.dtype), batch):
            return False
    return True


def _widened_integer_reduction(source: sparse.csr_array | sparse.csc_array, lower: int, upper: int) -> sparse.csr_array:
    """Sum integer duplicates in 64-bit arithmetic, then keep the source dtype or promote exactly.

    ``lower`` and ``upper`` bound every signed partial sum.
    """
    wide = np.dtype(np.int64 if source.dtype.kind == "i" else np.uint64)
    limits = np.iinfo(wide)
    if lower < limits.min or upper > limits.max:
        message = f"Integer duplicate sums can exceed {wide} arithmetic."
        raise ValueError(message)

    result = _compressed_to_csr(source.astype(wide), owns_data=True)
    minimum, maximum = int(result.data.min(initial=0)), int(result.data.max(initial=0))
    source_limits = np.iinfo(source.dtype)
    if source_limits.min <= minimum and maximum <= source_limits.max:
        dtype = source.dtype
    elif _exact_integer_cast(result.data, np.dtype(np.float32)):
        dtype = np.dtype(np.float32)
    elif _exact_integer_cast(result.data, np.dtype(np.float64)):
        dtype = np.dtype(np.float64)
    else:
        message = "Integer duplicate sums have no supported lossless float32 or float64 representation."
        raise ValueError(message)
    return sparse.csr_array((result.data.astype(dtype), result.indices, result.indptr), shape=result.shape, copy=False)


def normalize_sparse_duplicates(
    source: sparse.csr_array | sparse.csc_array | sparse.csr_matrix | sparse.csc_matrix,
) -> sparse.csr_array:
    """Return canonical CSR with exact completed integer coordinate totals.

    Parameters
    ----------
    source
        Borrowed in-memory CSR/CSC sparse array or matrix. Repeated coordinates
        are additive entries, including entries whose completed sum is zero.

    Returns
    -------
    scipy.sparse.csr_array
        Canonical CSR retaining sufficient source dtype. Integer sums outside
        the source dtype promote to float32 when every completed value is
        exactly representable, otherwise to float64 when exact. Suitable
        canonical CSR arrays are reused. Other conversions allocate as
        required without changing source buffers.

    Raises
    ------
    TypeError, ValueError
        If storage or values are unsupported, malformed or nonfinite, partial
        integer sums can exceed 64-bit arithmetic, or no supported floating
        dtype preserves the completed values on promotion.

    Notes
    -----
    A bound of the largest contribution magnitude times the longest
    compressed span decides whether integer duplicates can leave the source
    dtype; only then are they summed in int64 or uint64. Unaffected entries
    must also fit the promoted dtype. No expression is densified.
    Floating/boolean numerical semantics remain unchanged. This function
    neither infers count semantics nor emits warnings; readers select expected
    aggregation versus imported-data reporting.
    """
    _validate_value_dtype(source.dtype, parameter="data")
    validate_compressed_sparse(source)
    # A fresh wrapper recomputes canonicality from storage rather than trusting
    # cached flags that third-party buffers may have invalidated. It borrows
    # buffers; noncanonical reduction in as_csr_array establishes ownership.
    compressed = (
        sparse.csr_array((source.data, source.indices, source.indptr), shape=source.shape, copy=False)
        if source.format == "csr"
        else sparse.csc_array((source.data, source.indices, source.indptr), shape=source.shape, copy=False)
    )
    if compressed.has_canonical_format:
        if (
            isinstance(source, sparse.csr_array)
            and source.has_canonical_format
            and source.data.size == compressed.data.size
        ):
            return _compressed_to_csr(source, owns_data=False)
        return _compressed_to_csr(compressed, owns_data=False)
    if compressed.dtype.kind in "iu":
        span = _maximum_span(compressed.indptr)
        lower, upper = int(compressed.data.min(initial=0)) * span, int(compressed.data.max(initial=0)) * span
        limits = np.iinfo(compressed.dtype)
        if lower < limits.min or upper > limits.max:
            if not compressed.has_sorted_indices:
                # Sorting alone cannot overflow values. Unique coordinates
                # require no arithmetic bound, even at the integer extrema.
                compressed = compressed.copy()
                compressed.sort_indices()
                if compressed.has_canonical_format:
                    return _compressed_to_csr(compressed, owns_data=True)
            return _widened_integer_reduction(compressed, lower, upper)
    return _compressed_to_csr(compressed, owns_data=False)
