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
from spatialdata_io.readers._utils.table._count_values import count_accumulation_dtype

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


def _integer_span(source: sparse.csr_array | sparse.csc_array, start: int, stop: int) -> dict[int, int]:
    totals: dict[int, int] = {}
    for index, value in zip(source.indices[start:stop], source.data[start:stop], strict=True):
        key = int(index)
        totals[key] = totals.get(key, 0) + int(value)
    return totals


def _exact_integer_reduction(
    source: sparse.csr_array | sparse.csc_array, arithmetic_dtype: np.dtype[np.float32 | np.float64] | None
) -> sparse.csr_array:
    """Retain sufficient integers, otherwise select exact floating promotion."""
    minimum = maximum = entries = 0
    exact_float32 = exact_float64 = True
    check_float32 = arithmetic_dtype != np.dtype(np.float32)
    for start, stop in zip(source.indptr[:-1], source.indptr[1:], strict=True):
        totals = _integer_span(source, int(start), int(stop))
        entries += len(totals)
        for value in totals.values():
            minimum, maximum = min(minimum, value), max(maximum, value)
            if check_float32 and exact_float32:
                converted = np.float32(value)
                exact_float32 = bool(np.isfinite(converted)) and int(converted) == value
            if arithmetic_dtype is None:
                exact_float64 = exact_float64 and int(float(value)) == value
    limits = np.iinfo(source.dtype)
    if limits.min <= minimum and maximum <= limits.max:
        # Signed cancellation is permitted: fixed-width integer reduction has
        # the exact completed value whenever that value fits the source dtype.
        return _compressed_to_csr(source, owns_data=False)

    if not exact_float64:
        message = "Integer duplicate sums have no supported lossless float32 or float64 representation."
        raise ValueError(message)
    dtype = np.dtype(np.float32 if exact_float32 else np.float64)
    if arithmetic_dtype is not None:
        # astype casts before reducing duplicates. The magnitude bound proves
        # every source value and signed partial sum exact in this arithmetic.
        result = _compressed_to_csr(source.astype(arithmetic_dtype), owns_data=True)
        return (
            result
            if result.dtype == dtype
            else sparse.csr_array((result.data.astype(dtype), result.indices, result.indptr), shape=result.shape)
        )

    # Without a safe floating bound, build from completed Python-integer totals
    # rather than rounding the individual contributions before summation.
    values = np.empty(entries, dtype=dtype)
    indices = np.empty(entries, dtype=source.indices.dtype)
    indptr = np.empty_like(source.indptr)
    position = 0
    for major, (start, stop) in enumerate(zip(source.indptr[:-1], source.indptr[1:], strict=True)):
        indptr[major] = position
        totals = _integer_span(source, int(start), int(stop))
        for index in sorted(totals):
            indices[position] = index
            values[position] = totals[index]
            position += 1
    indptr[-1] = position
    result = (
        sparse.csr_array((values, indices, indptr), shape=source.shape)
        if source.format == "csr"
        else sparse.csc_array((values, indices, indptr), shape=source.shape)
    )
    return _compressed_to_csr(result, owns_data=True)


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
        Canonical CSR retaining sufficient source dtype. Integer overflow
        promotes directly to float32 when every completed value is exact,
        otherwise float64 if exact. Suitable canonical CSR arrays are reused. Other
        conversions allocate as required without changing source buffers.

    Raises
    ------
    TypeError, ValueError
        If storage or values are unsupported, malformed or nonfinite, or no
        supported floating dtype preserves every completed total on promotion.

    Notes
    -----
    Conservative bounds handle ordinary integer inputs; exceptional inputs
    use exact Python integers with workspace bounded by one compressed span.
    Safe floating bounds permit casting before SciPy reduction; otherwise the
    output is built from completed exact totals. No source contribution is
    rounded before addition and no wider integer array is constructed.
    Unaffected entries must also fit the promoted dtype.
    No expression is densified. Floating/boolean numerical semantics remain
    unchanged. This function neither infers count semantics nor emits warnings;
    readers select expected aggregation versus imported-data reporting.
    """
    validate_compressed_sparse(source)
    _validate_value_dtype(source.dtype, parameter="data")
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
        limits = np.iinfo(compressed.dtype)
        values = compressed.data
        maximum_records = 0
        for start in range(0, len(compressed.indptr) - 1, _BOUND_BATCH_SIZE):
            spans = np.diff(compressed.indptr[start : start + _BOUND_BATCH_SIZE + 1])
            maximum_records = max(maximum_records, int(spans.max(initial=0)))
        minimum, maximum = int(values.min(initial=0)), int(values.max(initial=0))
        if minimum * maximum_records < limits.min or maximum * maximum_records > limits.max:
            # Bounding sums of absolute contributions also bounds signed
            # partial sums, without changing the imported expression contract.
            arithmetic_dtype = count_accumulation_dtype(max(-minimum, maximum) * maximum_records)
            return _exact_integer_reduction(compressed, arithmetic_dtype)
    return _compressed_to_csr(compressed, owns_data=False)
