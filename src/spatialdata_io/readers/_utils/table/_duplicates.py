"""Lossless normalization of imported compressed sparse duplicate entries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

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


def _canonical_csr(data: sparse.csr_array | sparse.csc_array, *, owned: bool) -> sparse.csr_array:
    """Reduce only after integer arithmetic has been proved safe."""
    result = sparse.csr_array(
        data, copy=not owned and isinstance(data, sparse.csr_array) and not data.has_canonical_format
    )
    if not result.has_canonical_format:
        result.sum_duplicates()
    return result


def _check_duplicate_groups(source: sparse.csr_array | sparse.csc_array, wide: np.dtype) -> None:
    """Resolve an inconclusive global bound on the sorted coordinates that repeat."""
    limits = np.iinfo(wide)
    for begin, end in zip(source.indptr[:-1], source.indptr[1:], strict=True):
        indices = source.indices[begin:end]
        repeated = np.flatnonzero(indices[1:] == indices[:-1])
        if repeated.size == 0:
            continue
        for run in np.split(repeated, np.flatnonzero(np.diff(repeated) != 1) + 1):
            values = source.data[begin + run[0] : begin + run[-1] + 2]
            positive = sum(int(value) for value in values if value > 0)
            negative = sum(int(value) for value in values if value < 0)
            if negative < limits.min or positive > limits.max:
                message = f"Integer duplicate sums can exceed {wide} arithmetic."
                raise ValueError(message)


def _widened_integer_reduction(source: sparse.csr_array | sparse.csc_array, lower: int, upper: int) -> sparse.csr_array:
    """Sum integer duplicates in 64-bit arithmetic, then keep the source dtype or promote exactly.

    ``lower`` and ``upper`` bound every signed partial sum.
    """
    wide = np.dtype(np.int64 if source.dtype.kind == "i" else np.uint64)
    limits = np.iinfo(wide)
    if lower < limits.min or upper > limits.max:
        _check_duplicate_groups(source, wide)

    result = _canonical_csr(source.astype(wide), owned=True)
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
    return sparse.csr_array(
        (result.data.astype(dtype, copy=False), result.indices, result.indptr), shape=result.shape, copy=False
    )


def reduce_duplicates(compressed: sparse.csr_array | sparse.csc_array) -> sparse.csr_array:
    """Reduce a structurally validated compressed array without mutating borrowed buffers."""
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
                    return _canonical_csr(compressed, owned=True)
            return _widened_integer_reduction(compressed, lower, upper)
    return _canonical_csr(compressed, owned=False)
