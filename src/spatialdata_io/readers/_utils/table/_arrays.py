"""Canonical SciPy sparse-array construction without implicit computation."""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np
from scipy import sparse

_SUPPORTED_VALUE_DTYPES = frozenset(
    np.dtype(dtype)
    for dtype in (
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
        np.longdouble,
    )
)
_FINITE_VALIDATION_CHUNK_SIZE = 1_048_576
_MATRIX_DIMENSIONS = 2


class _ShapedArray(Protocol):
    """Small structural boundary around a NumPy or SciPy array shape."""

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """Return the array dtype."""

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the array shape."""


def _validate_value_dtype(dtype: np.dtype[np.generic], *, parameter: str) -> None:
    """Reject value dtypes that SciPy sparse arrays do not support."""
    if dtype not in _SUPPORTED_VALUE_DTYPES:
        message = (
            f"{parameter} must have a SciPy-supported boolean, integer, or real floating dtype; "
            f"found {dtype}. Float16, complex, object, structured, and string values are unsupported."
        )
        raise TypeError(message)


def _validate_finite_values(
    data: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    *,
    parameter: str,
) -> None:
    """Reject nonfinite floating values, scanning in bounded batches."""
    if data.dtype.kind != "f":
        return
    for start in range(0, data.size, _FINITE_VALIDATION_CHUNK_SIZE):
        values = data[start : start + _FINITE_VALIDATION_CHUNK_SIZE]
        if not np.isfinite(values).all():
            message = f"{parameter} must contain only finite values."
            raise ValueError(message)


def _validate_two_dimensions(data: _ShapedArray) -> None:
    """Require two-dimensional expression data."""
    if data.ndim != _MATRIX_DIMENSIONS:
        message = f"Expression data must be two-dimensional; found shape {data.shape}."
        raise ValueError(message)


def _canonicalize_csr(data: sparse.csr_array, *, owns_data: bool) -> sparse.csr_array:
    """Return ``data`` with duplicates summed and indices sorted.

    The input is mutated only when ``owns_data`` is true; otherwise a noncanonical input is copied first.
    """
    if data.has_canonical_format:
        return data
    result = data if owns_data else data.copy()
    result.sum_duplicates()
    return result


def _compressed_to_csr(
    data: sparse.csr_array | sparse.csc_array | sparse.csr_matrix | sparse.csc_matrix, *, owns_data: bool
) -> sparse.csr_array:
    """Canonicalize validated supported compressed storage and check finite results."""
    if isinstance(data, sparse.csr_array):
        result = _canonicalize_csr(data, owns_data=owns_data)
    else:
        copy = isinstance(data, sparse.csr_matrix) and not owns_data and not data.has_canonical_format
        result = sparse.csr_array(data, copy=copy)
        result = _canonicalize_csr(result, owns_data=True)
    _validate_finite_values(result.data, parameter="data")
    return result


def validate_compressed_sparse(
    data: sparse.csr_array | sparse.csc_array | sparse.csr_matrix | sparse.csc_matrix,
    /,
) -> None:
    """Validate borrowed two-dimensional CSR or CSC structure without mutation.

    Parameters
    ----------
    data
        In-memory SciPy CSR or CSC array or matrix.

    Raises
    ------
    ValueError
        If the shape, buffer dimensions, lengths, pointers or active indices
        violate the compressed sparse structure.

    Notes
    -----
    SciPy's ``check_format(full_check=True)`` runs on a fresh wrapper of the
    borrowed buffers, so its pruning and index-dtype normalization never
    change ``data`` and cached sorting or canonicality flags are not trusted.
    Unsorted indices and duplicates are valid. Data and index capacity beyond
    the terminal pointer is permitted and not inspected.
    """
    _validate_two_dimensions(data)
    wrapper = sparse.csr_array if data.format == "csr" else sparse.csc_array
    view = wrapper((data.data, data.indices, data.indptr), shape=data.shape, copy=False)
    view.check_format(full_check=True)
    # SciPy skips its pointer-order check without active entries; a terminal
    # pointer of at most zero is then only well formed when every pointer is zero.
    if int(view.indptr[-1]) <= 0 and view.indptr.any():
        message = "indptr must be nondecreasing from zero."
        raise ValueError(message)


def as_csr_array(
    data: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | sparse.sparray | sparse.spmatrix,
    /,
) -> sparse.csr_array:
    """Return finite two-dimensional expression data as a canonical CSR array.

    This function accepts plain NumPy arrays and any two-dimensional SciPy
    sparse array or sparse matrix. It deliberately rejects generic array-like,
    masked, backed, pandas, and lazy inputs so conversion cannot trigger hidden
    computation. SciPy-supported boolean, integer, and real floating dtypes are
    preserved; float16, complex, object, structured, string, and non-finite
    values are rejected.

    A canonical :class:`scipy.sparse.csr_array` is returned unchanged. A
    canonical CSR matrix can be wrapped without copying its storage. Other
    formats allocate CSR storage as required, and noncanonical CSR inputs are
    copied before their duplicate or index structures are changed. The return
    value can therefore share buffers with a canonical CSR input; callers that
    require independent mutable ownership must copy explicitly.

    Parameters
    ----------
    data
        A plain two-dimensional NumPy array or a two-dimensional SciPy sparse
        array or matrix.

    Returns
    -------
    scipy.sparse.csr_array
        Canonical CSR sparse-array data with the input shape and dtype.

    Raises
    ------
    TypeError
        If ``data`` is not a supported in-memory container or has an unsupported
        dtype.
    ValueError
        If ``data`` is not two-dimensional, contains non-finite values, or has
        malformed CSR/CSC storage.

    Notes
    -----
    Conversion is eager by design. Sparse inputs are never densified. Explicit
    stored zeros follow SciPy's canonical CSR semantics and are not removed.
    """
    if sparse.issparse(data):
        # scipy-stubs does not narrow issparse() to its concrete base classes
        # and omits metadata attributes from the sparray/spmatrix bases. Every
        # object accepted by scipy.sparse.issparse has this runtime contract.
        sparse_data = cast("sparse.sparray | sparse.spmatrix", data)
        sparse_metadata = cast("_ShapedArray", sparse_data)
        if isinstance(sparse_data, (sparse.csr_array, sparse.csc_array, sparse.csr_matrix, sparse.csc_matrix)):
            _validate_value_dtype(sparse_metadata.dtype, parameter="data")
            validate_compressed_sparse(sparse_data)
            return _compressed_to_csr(sparse_data, owns_data=False)
        _validate_two_dimensions(sparse_metadata)
        _validate_value_dtype(sparse_metadata.dtype, parameter="data")

        # scipy-stubs does not model the public sparse base classes as
        # accepted by csr_array, although SciPy supports this conversion.
        result = sparse.csr_array(sparse_data, copy=False)  # pyrefly: ignore[no-matching-overload]
        result = _canonicalize_csr(result, owns_data=True)
    elif type(data) is np.ndarray:
        _validate_two_dimensions(data)
        _validate_value_dtype(data.dtype, parameter="data")
        # Runtime dtype validation narrows the generic NumPy dtype to the
        # scalar families supported by the SciPy constructor.
        result = sparse.csr_array(
            cast(
                "np.ndarray[tuple[int, ...], np.dtype[np.bool_ | np.integer[Any] | np.floating[Any]]]",
                data,
            ),
            copy=False,
        )
    else:
        message = f"data must be a plain NumPy ndarray or a SciPy sparse array or matrix; found {type(data).__name__}."
        raise TypeError(message)

    _validate_finite_values(result.data, parameter="data")
    return result
