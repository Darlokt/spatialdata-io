"""Exact floating accumulation and storage selection for sparse counts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_array

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from pathlib import Path

_VALIDATION_BATCH_SIZE = 1_048_576
_FLOAT32_EXACT_INTEGERS = 2**24
_FLOAT64_EXACT_INTEGERS = 2**53
_FLOAT_STORAGE = frozenset((np.dtype(np.float32), np.dtype(np.float64)))


def count_accumulation_dtype(maximum_sum: int, /) -> np.dtype[np.float32 | np.float64] | None:
    """Select floating storage that holds every integer up to a nonnegative bound exactly.

    Parameters
    ----------
    maximum_sum
        Nonnegative Python-integer upper bound, either on every partial sum of
        an accumulation or on completed values.

    Returns
    -------
    numpy.dtype or None
        Float32 for bounds at most 2**24, otherwise float64 for bounds at most
        2**53. Larger bounds return ``None``: exact floating storage is
        unproven, and the caller chooses its fallback or contextual error.

    Notes
    -----
    This scalar decision performs no array scans, I/O, aggregation, or mutation.
    """
    if maximum_sum <= _FLOAT32_EXACT_INTEGERS:
        return np.dtype(np.float32)
    if maximum_sum <= _FLOAT64_EXACT_INTEGERS:
        return np.dtype(np.float64)
    return None


def _maximum_count(array: csr_array, context: ReaderErrorContext, path: Path | None) -> int:
    """Return the largest stored count after rejecting nonfinite, negative or fractional values."""
    maximum = 0
    for start in range(0, array.nnz, _VALIDATION_BATCH_SIZE):
        values = array.data[start : start + _VALIDATION_BATCH_SIZE]
        invalid = values.dtype.kind == "f" and not (
            np.isfinite(values).all() and np.array_equal(values, np.floor(values))
        )
        if invalid or np.any(values < 0):
            raise ReaderFormatError(
                context=context,
                path=path,
                found="nonfinite, negative or fractional counts",
                expected="finite nonnegative integer counts",
            )
        # Validated values are integral, so the conversion is exact.
        maximum = max(maximum, int(values.max(initial=0)))
    return maximum


def finalize_count_arrays(
    arrays: tuple[csr_array, ...],
    /,
    *,
    context: ReaderErrorContext,
    path: Path | None = None,
) -> tuple[csr_array, ...]:
    """Select one exact float32 or float64 dtype for completed owned counts.

    Parameters
    ----------
    arrays
        Nonempty tuple of reader-owned canonical CSR sparse arrays with
        boolean, integer or float32/float64 values. Complete all count
        arithmetic before calling; related arrays such as genes/blanks belong
        together.
    context
        Reader and artifact identity for invalid values or precision failures.
    path
        Optional reader-selected source artifact for numerical error context.

    Returns
    -------
    tuple of scipy.sparse.csr_array
        Arrays in input order, all float32 if the family maximum is at most
        2**24, otherwise all float64 if it is at most 2**53. A family stored
        entirely in one floating dtype is never widened: its stored integral
        values are already exact. Already suitable arrays are returned
        unchanged. Converted arrays own new values and may share the owned
        canonical structural buffers; do not mutate shared structures.

    Raises
    ------
    ReaderFormatError
        If counts are nonfinite, negative or fractional, or exceed 2**53
        without already sharing one floating storage dtype.

    Notes
    -----
    Stored-value validation uses bounded temporary buffers, never dense
    expression storage. Explicit zeros, shapes, row/feature order and index
    widths are preserved. No input is mutated and duplicates are not summed.
    Exactness is relative to the supplied values, including values decoded
    upstream into floating storage.
    """
    maximum = max(_maximum_count(array, context, path) for array in arrays)
    dtype = count_accumulation_dtype(maximum)
    stored = {array.dtype for array in arrays}
    if len(stored) == 1 and stored <= _FLOAT_STORAGE:
        # Stored floating values are already exact; widening would add memory, not precision.
        (existing,) = stored
        dtype = existing if dtype is None or dtype.itemsize > existing.itemsize else dtype
    if dtype is None:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="counts above 2**53 in integer or mixed storage",
            expected="counts at most 2**53, or a family already stored in one floating dtype",
        )
    return tuple(
        array
        if array.dtype == dtype
        else csr_array((array.data.astype(dtype), array.indices, array.indptr), shape=array.shape, copy=False)
        for array in arrays
    )
