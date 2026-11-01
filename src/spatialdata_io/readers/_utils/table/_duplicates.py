"""Lossless normalization of imported compressed sparse duplicate entries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from spatialdata_io.readers._utils.table._arrays import as_csr_array, validate_compressed_sparse


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


def _integer_dtype(
    source_dtype: np.dtype[np.integer], minimum: int, maximum: int, *, exact_float64: bool
) -> np.dtype[np.integer | np.float64]:
    limits = np.iinfo(source_dtype)
    if limits.min <= minimum and maximum <= limits.max:
        return source_dtype
    candidates = (np.int32, np.int64, np.uint64) if source_dtype.kind == "i" else (np.uint32, np.uint64)
    for candidate in candidates:
        dtype = np.dtype(candidate)
        limits = np.iinfo(dtype)
        if dtype.itemsize >= source_dtype.itemsize and limits.min <= minimum and maximum <= limits.max:
            return dtype
    if exact_float64:
        return np.dtype(np.float64)
    message = "Integer duplicate sums have no supported lossless integer or float64 representation."
    raise ValueError(message)


def _exact_integer_reduction(source: sparse.csr_array | sparse.csc_array) -> sparse.csr_array:
    """Analyze and build exact totals with only one compressed span in Python."""
    minimum = maximum = entries = 0
    exact_float64 = True
    for start, stop in zip(source.indptr[:-1], source.indptr[1:], strict=True):
        totals = _integer_span(source, int(start), int(stop))
        entries += len(totals)
        for value in totals.values():
            minimum, maximum = min(minimum, value), max(maximum, value)
            exact_float64 = exact_float64 and int(float(value)) == value
    dtype = _integer_dtype(source.dtype, minimum, maximum, exact_float64=exact_float64)
    if dtype == source.dtype:
        # Signed cancellation is permitted: fixed-width integer reduction has
        # the exact completed value whenever that value fits the source dtype.
        return as_csr_array(source)

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
    return as_csr_array(result)


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
        Canonical CSR retaining sufficient source dtype. Integer values promote
        losslessly when necessary; float64 fallback requires every completed
        value to be exact. Suitable canonical CSR arrays are reused. Other
        conversions allocate as required without changing source buffers.

    Raises
    ------
    TypeError, ValueError
        If storage or values are unsupported, malformed or nonfinite, or no
        supported integer or float64 dtype preserves every completed total.

    Notes
    -----
    Conservative bounds handle ordinary integer inputs; exceptional inputs
    use exact Python integers with workspace bounded by one compressed span.
    No expression is densified. Floating/boolean numerical semantics remain
    unchanged. This function neither infers count semantics nor emits warnings;
    readers select expected aggregation versus imported-data reporting.
    """
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
        if isinstance(source, sparse.csr_array) and source.has_canonical_format:
            return as_csr_array(source)
        return as_csr_array(compressed)
    if compressed.dtype.kind in "iu":
        limits = np.iinfo(compressed.dtype)
        values = compressed.data
        records = int(compressed.indptr[-1])
        if int(values.min(initial=0)) * records < limits.min or int(values.max(initial=0)) * records > limits.max:
            return _exact_integer_reduction(compressed)
    return as_csr_array(compressed)
