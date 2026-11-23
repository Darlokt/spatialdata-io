"""Canonical SciPy sparse-array construction without implicit computation."""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np
from scipy import sparse

from spatialdata_io.readers._utils.table._duplicates import reduce_duplicates

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


def _validate_compressed_sparse(
    data: sparse.csr_array | sparse.csc_array | sparse.csr_matrix | sparse.csc_matrix,
    /,
) -> sparse.csr_array | sparse.csc_array:
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
    The SciPy constructor checks the buffer shapes, endpoints and capacities
    on a fresh wrapper. Compare adjacent pointers and check the active indices
    before any numerical operation. The caller retains its original buffers.
    Unsorted indices and duplicates are valid. Data and index capacity beyond
    the terminal pointer is permitted and not inspected.
    """
    _validate_two_dimensions(data)
    if data.indices.dtype.kind not in "iu" or data.indptr.dtype.kind not in "iu":
        message = "Compressed sparse indices and pointers must have integer dtypes."
        raise ValueError(message)
    wrapper = sparse.csr_array if data.format == "csr" else sparse.csc_array
    view = wrapper((data.data, data.indices, data.indptr), shape=data.shape, copy=False)
    # Comparing adjacent pointers directly works at signed integer extrema and
    # also detects invalid intervals when the terminal pointer is zero.
    if np.any(view.indptr[1:] < view.indptr[:-1]):
        message = "indptr must be nondecreasing from zero."
        raise ValueError(message)
    minor_size = view.shape[1] if data.format == "csr" else view.shape[0]
    if view.indices.size:
        if view.indices.min() < 0:
            message = "indices must be >= 0"
            raise ValueError(message)
        if view.indices.max() >= minor_size:
            message = f"indices must be < {minor_size}"
            raise ValueError(message)
    return view


def _sparse_expression(data: sparse.sparray | sparse.spmatrix) -> sparse.csr_array:
    """Validate borrowed storage before any conversion can combine coordinates."""
    # scipy-stubs omits shape, ndim and dtype from the abstract sparse bases.
    metadata = cast("_ShapedArray", data)
    _validate_value_dtype(metadata.dtype, parameter="data")
    if isinstance(data, (sparse.csr_array, sparse.csc_array, sparse.csr_matrix, sparse.csc_matrix)):
        compressed = _validate_compressed_sparse(data)
        if compressed.has_canonical_format:
            if (
                isinstance(data, sparse.csr_array)
                and data.has_canonical_format
                and data.data.size == compressed.data.size
                and data.indices.dtype == compressed.indices.dtype
                and data.indptr.dtype == compressed.indptr.dtype
            ):
                return data
            return sparse.csr_array(compressed)
    else:
        _validate_two_dimensions(metadata)
        # Coordinate-form construction of CSR would sum duplicates in the input dtype.
        # Group by row without arithmetic, preserving entries for the exact reducer.
        # scipy-stubs omits the constructor overload for the abstract sparse bases.
        coo = sparse.coo_array(data)  # pyrefly: ignore[no-matching-overload]
        values, indices = coo.data, coo.col
        if np.any(coo.row[1:] < coo.row[:-1]):
            order = np.argsort(coo.row, kind="stable")
            values, indices = values[order], indices[order]
        pointers = np.empty(coo.shape[0] + 1, dtype=np.intp)
        pointers[0] = 0
        np.cumsum(np.bincount(coo.row, minlength=coo.shape[0]), out=pointers[1:])
        compressed = sparse.csr_array((values, indices, pointers), shape=coo.shape)
        if compressed.has_canonical_format:
            return compressed
    return reduce_duplicates(compressed)


def as_csr_array(
    data: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | sparse.sparray | sparse.spmatrix,
    /,
) -> sparse.csr_array:
    """Return finite two-dimensional expression data as a canonical CSR array.

    This function accepts plain NumPy arrays and any two-dimensional SciPy
    sparse array or sparse matrix. It deliberately rejects generic array-like,
    masked, backed, pandas, and lazy inputs so conversion cannot trigger hidden
    computation. SciPy-supported boolean, integer, and real floating dtypes are
    preserved when duplicate sums fit. Integer overflow requires exact promotion.
    Float16, complex, object, structured, string, and non-finite values are rejected.

    A canonical :class:`scipy.sparse.csr_array` is returned unchanged. A
    canonical CSR matrix can be wrapped without copying its storage. Other
    formats allocate CSR storage as required, and noncanonical CSR inputs are
    copied before their duplicate or index structures are changed. The return
    value can share buffers with canonical CSR or row-ordered COO input.
    Callers requiring independent mutable ownership must copy explicitly.

    Parameters
    ----------
    data
        A plain two-dimensional NumPy array or a two-dimensional SciPy sparse
        array or matrix.

    Returns
    -------
    scipy.sparse.csr_array
        Canonical CSR sparse-array data with the input shape and a lossless dtype for integer sums.

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
        result = _sparse_expression(cast("sparse.sparray | sparse.spmatrix", data))
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
