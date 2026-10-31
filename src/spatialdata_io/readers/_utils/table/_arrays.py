"""Canonical SciPy sparse-array construction without implicit computation."""

from __future__ import annotations

from numbers import Integral
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
_SUPPORTED_INDEX_DTYPES = frozenset((np.dtype(np.int32), np.dtype(np.int64)))
_FINITE_VALIDATION_CHUNK_SIZE = 1_048_576
_MATRIX_DIMENSIONS = 2


class _ShapedArray(Protocol):
    """Small structural boundary around a NumPy or SciPy array shape."""

    @property
    def dtype(self) -> np.dtype[np.generic]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def shape(self) -> tuple[int, ...]: ...


def _validate_value_dtype(dtype: np.dtype[np.generic], *, parameter: str) -> None:
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
    if data.dtype.kind != "f":
        return
    for start in range(0, data.size, _FINITE_VALIDATION_CHUNK_SIZE):
        values = data[start : start + _FINITE_VALIDATION_CHUNK_SIZE]
        if not np.isfinite(values).all():
            message = f"{parameter} must contain only finite values."
            raise ValueError(message)


def _validate_two_dimensions(data: _ShapedArray) -> None:
    if data.ndim != _MATRIX_DIMENSIONS:
        message = f"Expression data must be two-dimensional; found shape {data.shape}."
        raise ValueError(message)


def _canonicalize_csr(data: sparse.csr_array, *, owns_data: bool) -> sparse.csr_array:
    if data.has_canonical_format:
        return data
    result = data if owns_data else data.copy()
    result.sum_duplicates()
    return result


def validate_compressed_sparse(
    data: sparse.csr_array | sparse.csc_array | sparse.csr_matrix | sparse.csc_matrix,
    /,
) -> None:
    """Validate borrowed two-dimensional CSR or CSC storage without mutation.

    Parameters
    ----------
    data
        In-memory SciPy CSR or CSC array or matrix. Its data, indices and row
        or column pointers must be one-dimensional NumPy arrays. Index and
        pointer storage must have integer dtypes safely representable by
        signed int64, as required by SciPy's compressed sparse kernels.

    Raises
    ------
    TypeError
        If the container, buffers, index dtypes or shape are unsupported.
    ValueError
        If the shape, buffer dimensions, lengths, pointers or active indices
        violate the compressed sparse structure.

    Notes
    -----
    This eager, read-only inspection does not canonicalize, inspect expression
    values, call SciPy's mutating ``check_format()``, or trust cached sorting
    and canonicality flags. Unsorted indices and duplicates are valid. Equal
    data/index capacity beyond the terminal pointer is permitted; unused
    indices are not inspected. Scans use bounded temporary boolean arrays and
    never copy expression storage or construct another sparse object.
    """
    if not isinstance(data, (sparse.csr_array, sparse.csc_array, sparse.csr_matrix, sparse.csc_matrix)):
        message = f"data must be a SciPy CSR or CSC array or matrix; found {type(data).__name__}."
        raise TypeError(message)
    _validate_two_dimensions(data)
    shape = _validate_shape(data.shape)
    for name, buffer in (("data", data.data), ("indices", data.indices), ("indptr", data.indptr)):
        if not isinstance(buffer, np.ndarray):
            message = f"{name} must be a NumPy ndarray; found {type(buffer).__name__}."
            raise TypeError(message)
        if buffer.ndim != 1:
            message = f"{name} must be one-dimensional; found shape {buffer.shape}."
            raise ValueError(message)
    for name, indices in (("indices", data.indices), ("indptr", data.indptr)):
        if indices.dtype.kind not in "iu" or not np.can_cast(indices.dtype, np.int64, casting="safe"):
            message = f"{name} must have an integer dtype safely representable by int64; found {indices.dtype}."
            raise TypeError(message)
    if data.data.size != data.indices.size:
        message = f"data and indices must have equal lengths; found {data.data.size} and {data.indices.size}."
        raise ValueError(message)
    major_size, minor_size = shape if isinstance(data, (sparse.csr_array, sparse.csr_matrix)) else shape[::-1]
    terminal = _validate_compressed_pointers(data.indptr, major_size=major_size, capacity=data.data.size)
    for start in range(0, terminal, _FINITE_VALIDATION_CHUNK_SIZE):
        indices = data.indices[start : min(start + _FINITE_VALIDATION_CHUNK_SIZE, terminal)]
        if int(indices.min()) < 0 or int(indices.max()) >= minor_size:
            message = f"active indices must be in [0, {minor_size})."
            raise ValueError(message)


def validate_integer_duplicate_sums(
    data: sparse.csr_array | sparse.csc_array | sparse.csr_matrix | sparse.csc_matrix,
    /,
) -> None:
    """Validate that compressed integer duplicate sums retain their source dtype.

    Parameters
    ----------
    data
        Borrowed two-dimensional SciPy CSR or CSC array or matrix.

    Raises
    ------
    TypeError
        If the container or compressed storage types are unsupported.
    ValueError
        If compressed storage is malformed or a completed integer coordinate
        sum lies outside the source dtype's range.

    Notes
    -----
    Storage is validated before inspecting canonicality. Noninteger inputs and
    canonical integer inputs return without scanning expression magnitudes.
    A conservative min/max bound handles ordinary small noncanonical counts;
    otherwise exact Python integers accumulate each compressed span. Signed
    cancellation is allowed: only completed coordinate sums determine safety.

    This eager check never widens, converts through floating point, densifies,
    canonicalizes, or mutates borrowed buffers. The fallback uses a dictionary
    proportional to the distinct active indices in one row or column. Floating
    and boolean expression policies remain the caller's responsibility.
    """
    validate_compressed_sparse(data)
    if data.dtype.kind not in "iu" or data.has_canonical_format:
        return

    limits = np.iinfo(data.dtype)
    records = int(data.indptr[-1])
    values = data.data[:records]
    # Bound every possible coordinate sum without visiting records in Python.
    if int(values.min(initial=0)) * records >= limits.min and int(values.max(initial=0)) * records <= limits.max:
        return

    for start, stop in zip(data.indptr[:-1], data.indptr[1:], strict=True):
        totals: dict[int, int] = {}
        for index, value in zip(data.indices[start:stop], data.data[start:stop], strict=True):
            key = int(index)
            totals[key] = totals.get(key, 0) + int(value)
        if any(total < limits.min or total > limits.max for total in totals.values()):
            message = f"integer duplicate sum outside {data.dtype}; sums must be representable in the source dtype."
            raise ValueError(message)


def _validate_compressed_pointers(
    indptr: np.ndarray[tuple[int, ...], np.dtype[np.integer[Any]]],
    *,
    major_size: int,
    capacity: int,
) -> int:
    """Check the compressed span partition without subtraction or mutation."""
    if indptr.size != major_size + 1:
        message = f"indptr must have length {major_size + 1}; found {indptr.size}."
        raise ValueError(message)
    if int(indptr[0]) != 0:
        message = "indptr must start at zero."
        raise ValueError(message)
    terminal = int(indptr[-1])
    if terminal < 0 or terminal > capacity:
        message = f"indptr terminal extent must be in [0, {capacity}]; found {terminal}."
        raise ValueError(message)
    for start in range(0, major_size, _FINITE_VALIDATION_CHUNK_SIZE):
        stop = min(start + _FINITE_VALIDATION_CHUNK_SIZE, major_size)
        # Adjacent comparisons cannot overflow, unlike subtracting pointers.
        if np.any(indptr[start + 1 : stop + 1] < indptr[start:stop]):
            message = "indptr must be nondecreasing."
            raise ValueError(message)
    return terminal


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
            validate_compressed_sparse(sparse_data)
        else:
            _validate_two_dimensions(sparse_metadata)
        _validate_value_dtype(sparse_metadata.dtype, parameter="data")

        if isinstance(sparse_data, sparse.csr_array):
            result = _canonicalize_csr(sparse_data, owns_data=False)
        elif isinstance(sparse_data, sparse.csr_matrix):
            result = sparse.csr_array(sparse_data, copy=not sparse_data.has_canonical_format)
            result = _canonicalize_csr(result, owns_data=True)
        else:
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


def _validate_triplet_array(
    data: object,
    *,
    parameter: str,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    if type(data) is not np.ndarray:
        message = f"{parameter} must be a plain NumPy ndarray; found {type(data).__name__}."
        raise TypeError(message)
    if data.ndim != 1:
        message = f"{parameter} must be one-dimensional; found shape {data.shape}."
        raise ValueError(message)
    return data


def _validate_indices(
    indices: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    *,
    parameter: str,
) -> np.dtype[np.signedinteger[Any]]:
    if indices.dtype not in _SUPPORTED_INDEX_DTYPES:
        message = f"{parameter} must have dtype int32 or int64; found {indices.dtype}."
        raise TypeError(message)
    return cast("np.dtype[np.signedinteger[Any]]", indices.dtype)


def _validate_shape(shape: object) -> tuple[int, int]:
    if not isinstance(shape, tuple) or len(shape) != _MATRIX_DIMENSIONS:
        message = f"shape must be a pair of nonnegative integers; found {shape!r}."
        raise TypeError(message)

    normalized: list[int] = []
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, Integral):
            message = f"shape must be a pair of nonnegative integers; found {shape!r}."
            raise TypeError(message)
        if dimension < 0:
            message = f"shape dimensions must be nonnegative; found {shape!r}."
            raise ValueError(message)
        normalized.append(int(dimension))
    return normalized[0], normalized[1]


def _validate_index_bounds(
    indices: np.ndarray[tuple[int, ...], np.dtype[np.signedinteger[Any]]],
    *,
    size: int,
    parameter: str,
) -> None:
    if indices.size == 0:
        return
    minimum = int(indices.min())
    maximum = int(indices.max())
    if minimum < 0 or maximum >= size:
        message = f"{parameter} values must be in [0, {size}); found range [{minimum}, {maximum}]."
        raise ValueError(message)


def _validate_shape_for_index_dtype(shape: tuple[int, int], *, dtype: np.dtype[np.signedinteger[Any]]) -> None:
    maximum = np.iinfo(dtype).max
    if any(dimension > maximum for dimension in shape):
        message = f"shape {shape!r} cannot be represented by {dtype} indices."
        raise ValueError(message)


def csr_array_from_triplets(
    values: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    row_indices: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    column_indices: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    *,
    shape: tuple[int, int],
) -> sparse.csr_array:
    """Construct a canonical CSR array from explicit coordinate triplets.

    Duplicate coordinates are summed by SciPy before the result is returned.
    Values retain their SciPy-supported boolean, integer, or real floating dtype
    and must be finite. Row and column arrays must use the same signed ``int32``
    or ``int64`` dtype, which is preserved in the returned CSR structure. All
    inputs remain unmodified, including when duplicate summation or index sorting
    is required.

    Parameters
    ----------
    values
        One-dimensional values stored at the corresponding coordinates.
    row_indices
        One-dimensional zero-based observation indices.
    column_indices
        One-dimensional zero-based feature indices.
    shape
        Explicit ``(n_observations, n_features)`` result shape. Empty
        dimensions are supported only when no triplets address them.

    Returns
    -------
    scipy.sparse.csr_array
        Canonical CSR sparse array with duplicate coordinates summed.

    Raises
    ------
    TypeError
        If an input is not a plain NumPy array, values use an unsupported dtype,
        indices are not matching ``int32`` or ``int64`` arrays, or ``shape`` is
        not an integer pair.
    ValueError
        If inputs are not one-dimensional, have different lengths, contain
        non-finite values, use negative or structurally unrepresentable shape
        dimensions, or address an index outside ``shape``.

    Notes
    -----
    Duplicate summation preserves the value dtype. Readers must select a dtype
    wide enough for the largest valid aggregate.
    """
    values = _validate_triplet_array(values, parameter="values")
    row_indices = _validate_triplet_array(row_indices, parameter="row_indices")
    column_indices = _validate_triplet_array(column_indices, parameter="column_indices")
    _validate_value_dtype(values.dtype, parameter="values")
    row_index_dtype = _validate_indices(row_indices, parameter="row_indices")
    column_index_dtype = _validate_indices(column_indices, parameter="column_indices")

    if row_index_dtype != column_index_dtype:
        message = (
            "row_indices and column_indices must have the same dtype; "
            f"found {row_index_dtype} and {column_index_dtype}."
        )
        raise TypeError(message)

    if not (values.size == row_indices.size == column_indices.size):
        message = (
            "values, row_indices, and column_indices must have equal lengths; "
            f"found {values.size}, {row_indices.size}, and {column_indices.size}."
        )
        raise ValueError(message)

    normalized_shape = _validate_shape(shape)
    _validate_shape_for_index_dtype(normalized_shape, dtype=row_index_dtype)
    _validate_finite_values(values, parameter="values")

    # The runtime dtype checks above establish the narrower NumPy scalar
    # families required by scipy-stubs without copying the input arrays.
    numeric_values = cast(
        "np.ndarray[tuple[int, ...], np.dtype[np.bool_ | np.integer[Any] | np.floating[Any]]]",
        values,
    )
    integer_rows = cast(
        "np.ndarray[tuple[int, ...], np.dtype[np.signedinteger[Any]]]",
        row_indices,
    )
    integer_columns = cast(
        "np.ndarray[tuple[int, ...], np.dtype[np.signedinteger[Any]]]",
        column_indices,
    )

    _validate_index_bounds(integer_rows, size=normalized_shape[0], parameter="row_indices")
    _validate_index_bounds(integer_columns, size=normalized_shape[1], parameter="column_indices")

    result = sparse.csr_array(
        (numeric_values, (integer_rows, integer_columns)),
        shape=normalized_shape,
        copy=False,
    )
    _validate_finite_values(result.data, parameter="summed values")
    return result
