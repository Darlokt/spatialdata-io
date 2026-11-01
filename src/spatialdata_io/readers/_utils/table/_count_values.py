"""Exact floating accumulation and storage selection for sparse counts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_array

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.table._arrays import validate_compressed_sparse

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

_VALIDATION_BATCH_SIZE = 1_048_576
_UINT64_STOP = 2**64


def count_accumulation_dtype(maximum_sum: int, /) -> np.dtype[np.float32 | np.float64] | None:
    """Select floating arithmetic from a nonnegative integer partial-sum bound.

    Parameters
    ----------
    maximum_sum
        Nonnegative Python-integer upper bound on every partial sum. The
        reader validates finite nonnegative integral contributions and computes
        this bound without fixed-width overflow before calling.

    Returns
    -------
    numpy.dtype or None
        Float32 for bounds at most 2**24, otherwise float64 for bounds at most
        2**53. Larger bounds return ``None``: exact floating accumulation is
        unproven, and the reader chooses its fallback or contextual error.

    Raises
    ------
    TypeError
        If the bound is not a Python integer or is a boolean.
    ValueError
        If the bound is negative.

    Notes
    -----
    This scalar decision performs no array scans, I/O, aggregation, or mutation.
    It protects every partial sum rather than selecting storage for actual
    completed values; use :func:`finalize_count_arrays` for the latter.
    """
    if isinstance(maximum_sum, bool) or not isinstance(maximum_sum, int):
        message = "maximum_sum must be a nonnegative Python integer."
        raise TypeError(message)
    if maximum_sum < 0:
        message = "maximum_sum must be nonnegative."
        raise ValueError(message)
    if maximum_sum <= 2**24:
        return np.dtype(np.float32)
    if maximum_sum <= 2**53:
        return np.dtype(np.float64)
    return None


def _exact_float_cast(values: npt.NDArray[np.generic], dtype: np.dtype[np.float32 | np.float64]) -> bool:
    """Compare integers without first rounding both sides into floating storage."""
    if values.dtype.kind == "f" and values.dtype.itemsize <= dtype.itemsize:
        # Validation already established finite integral counts. Existing
        # floating storage embeds exactly in an equal/wider floating dtype.
        return True
    with np.errstate(over="ignore", invalid="ignore"):
        converted = values.astype(dtype)
    if not np.isfinite(converted).all():
        return False
    if values.dtype.kind in "iu":
        # Counts have already been proved nonnegative. The exclusive bound is
        # exactly representable in either float; never cast 2**64 to uint64.
        if np.any(converted >= _UINT64_STOP):
            return False
        return np.array_equal(converted.astype(np.uint64), values.astype(np.uint64, copy=False))
    return np.array_equal(converted, values)


def _validate_count_storage(array: csr_array) -> None:
    if not isinstance(array, csr_array):
        message = "Completed counts must be CSR sparse arrays."
        raise TypeError(message)
    validate_compressed_sparse(array)
    if not array.has_canonical_format:
        message = "Complete duplicate aggregation before count finalization."
        raise ValueError(message)
    if array.dtype.kind not in "biuf":
        message = f"Counts require boolean, integer or real floating values; found {array.dtype}."
        raise TypeError(message)


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
        Nonempty tuple of reader-owned canonical CSR sparse arrays. Counts
        must be finite, nonnegative and integral. Complete all count arithmetic
        before calling; related arrays such as genes/blanks belong together.
    context
        Reader and artifact identity for invalid values or precision failures.
    path
        Optional reader-selected source artifact for numerical error context.

    Returns
    -------
    tuple of scipy.sparse.csr_array
        Arrays in input order, all float32 if every value is exactly
        representable, otherwise all float64. Already suitable arrays are
        returned unchanged. Converted arrays own new values and may share the
        owned canonical structural buffers; do not mutate shared structures.

    Raises
    ------
    TypeError
        If an input is not a CSR sparse array or uses unsupported values.
    ValueError
        If the family is empty, sparse storage is malformed or not canonical.
    ReaderFormatError
        If counts are invalid or neither floating dtype preserves all values.

    Notes
    -----
    Stored-value validation uses bounded temporary buffers, never dense
    expression storage. Explicit zeros, shapes, row/feature order and index
    widths are preserved. No input is mutated and duplicates are not summed.
    Exactness is relative to the supplied values, including values decoded
    upstream into floating storage.
    """
    if not arrays:
        message = "A count family must contain at least one CSR array."
        raise ValueError(message)
    fits_float32 = True
    fits_float64 = True
    for array in arrays:
        _validate_count_storage(array)
        for start in range(0, array.nnz, _VALIDATION_BATCH_SIZE):
            values = array.data[start : start + _VALIDATION_BATCH_SIZE]
            if (
                not np.isfinite(values).all()
                or np.any(values < 0)
                or (values.dtype.kind == "f" and np.any(values != np.floor(values)))
            ):
                raise ReaderFormatError(
                    context=context,
                    path=path,
                    found="nonfinite, negative or fractional counts",
                    expected="finite nonnegative integer counts",
                )
            if fits_float32:
                fits_float32 = _exact_float_cast(values, np.dtype(np.float32))
            if fits_float64:
                fits_float64 = _exact_float_cast(values, np.dtype(np.float64))
    if not fits_float64:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="counts not exactly representable in float64",
            expected="completed counts exactly representable in float32 or float64",
        )
    dtype = np.dtype(np.float32 if fits_float32 else np.float64)
    return tuple(
        array
        if array.dtype == dtype
        else csr_array((array.data.astype(dtype), array.indices, array.indptr), shape=array.shape, copy=False)
        for array in arrays
    )
