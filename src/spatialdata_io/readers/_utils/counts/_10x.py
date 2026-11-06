"""Typed public-Scanpy boundaries for 10x count containers."""

from __future__ import annotations

import gzip
from typing import TYPE_CHECKING

import numpy as np
from scanpy import read_10x_h5 as _scanpy_read_10x_h5
from scanpy import read_10x_mtx as _scanpy_read_10x_mtx
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.table import (
    count_accumulation_dtype,
    normalize_owned_anndata,
    validate_compressed_sparse,
)
from spatialdata_io.readers._utils.table._duplicates import maximum_span

if TYPE_CHECKING:
    from pathlib import Path

    from anndata import AnnData
    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile

_VALIDATION_BATCH_SIZE = 1_048_576


def _prepare_count_arithmetic(
    matrix: sparse.csr_array | sparse.csc_array | sparse.csr_matrix | sparse.csc_matrix,
    path: Path,
    context: ReaderErrorContext,
) -> sparse.csr_array | sparse.csc_array | sparse.csr_matrix | sparse.csc_matrix:
    """Choose exact arithmetic before reducing decoded floating or boolean counts."""
    # Recompute canonicality without changing a borrowed object's cached flags.
    source = (
        sparse.csr_array((matrix.data, matrix.indices, matrix.indptr), shape=matrix.shape, copy=False)
        if isinstance(matrix, (sparse.csr_array, sparse.csr_matrix))
        else sparse.csc_array((matrix.data, matrix.indices, matrix.indptr), shape=matrix.shape, copy=False)
    )
    if source.has_canonical_format:
        return matrix

    owns_buffers = False
    if not source.has_sorted_indices:
        source = source.copy()
        source.sort_indices()
        owns_buffers = True
        if source.has_canonical_format:
            return source

    bound = int(source.data.max(initial=0)) * maximum_span(source.indptr)
    dtype = count_accumulation_dtype(bound)
    if dtype is None:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="a decoded duplicate-sum bound exceeding float64 consecutive-integer precision",
            expected="an exact floating duplicate-sum bound at most 2**53",
        )
    if matrix.dtype.itemsize < dtype.itemsize:
        # SciPy casts values before combining duplicates. Its dtype-changing
        # conversion also owns the structural buffers needed by the reduction.
        return source.astype(dtype)
    if owns_buffers:
        source.sum_duplicates()
    return source


def _validate_count_values(
    values: NDArray[np.bool_ | np.integer | np.floating], path: Path, context: ReaderErrorContext
) -> None:
    """Validate retained count contributions before duplicate reduction."""
    if values.size and values.min() < 0:
        raise ReaderFormatError(
            context=context, path=path, found="negative counts", expected="nonnegative count values"
        )
    if values.dtype.kind != "f":
        return
    for start in range(0, values.size, _VALIDATION_BATCH_SIZE):
        batch = values[start : start + _VALIDATION_BATCH_SIZE]
        if not np.isfinite(batch).all():
            raise ReaderFormatError(
                context=context, path=path, found="nonfinite counts", expected="finite count values"
            )
        if np.any(batch != np.floor(batch)):
            raise ReaderFormatError(
                context=context, path=path, found="fractional counts", expected="integer count values"
            )


def _normalize_counts(adata: AnnData, path: Path, context: ReaderErrorContext) -> AnnData:
    """Validate the Scanpy count matrix and return ``adata`` normalized to canonical CSR.

    Entries are inspected before normalization so that summing duplicates cannot hide
    a negative or fractional source value.
    """
    matrix = adata.X
    if matrix is None:
        raise ReaderFormatError(context=context, path=path, found="missing count data", expected="a count matrix")
    # Scanpy returns CSR (HDF5) or CSC (MTX). Inspect entries before normalization
    # can sum duplicates and hide a negative or fractional source value.
    if not isinstance(matrix, sparse.csr_array | sparse.csr_matrix | sparse.csc_array | sparse.csc_matrix) or (
        matrix.dtype.kind not in "biuf"
    ):
        message = "The 10x source must return a real numerical CSR or CSC count matrix."
        raise TypeError(message)
    if matrix.dtype == np.dtype(np.float16):
        message = "Float16 count storage is unsupported."
        raise TypeError(message)
    validate_compressed_sparse(matrix)
    values = matrix.data[: int(matrix.indptr[-1])]
    _validate_count_values(values, path, context)
    if matrix.dtype.kind in "bf":
        adata.X = _prepare_count_arithmetic(matrix, path, context)
    return normalize_owned_anndata(adata)


def _validate_mtx_header(
    path: ArtifactDirectory, *, context: ReaderErrorContext, prefix: str | None, compressed: bool
) -> None:
    """Check only the type declaration; Scanpy owns all record parsing."""
    stem = prefix or ""
    # Match Scanpy's legacy family selection, including its uncompressed v2 inputs.
    legacy = (path / f"{stem}genes.tsv").is_file()
    zipped = compressed and not legacy
    file = path / f"{stem}matrix.mtx{'.gz' if zipped else ''}"
    with gzip.open(file, "rb") if zipped else file.open("rb") as source:
        header = source.readline()
    fields = header.lower().split()
    if fields[:-1] != [b"%%matrixmarket", b"matrix", b"coordinate", b"integer"] or fields[-1:] not in (
        [b"general"],
        [b"symmetric"],
    ):
        raise ReaderFormatError(
            context=context,
            path=file,
            found=f"MatrixMarket header {header!r}",
            expected="an integer coordinate matrix with general or symmetric storage",
        )


def read_10x_h5(path: ArtifactFile, /, *, context: ReaderErrorContext) -> AnnData:
    """Read one validated 10x HDF5 count container into owned sparse storage.

    Parameters
    ----------
    path
        Canonical regular file previously validated by the shared path
        boundary. This function borrows the path and does not validate or
        reopen it outside Scanpy.
    context
        Reader and artifact identity added to failures without changing their
        exception type or traceback.

    Returns
    -------
    anndata.AnnData
        An owned, eager, in-memory table whose ``X`` is a canonical
        :class:`scipy.sparse.csr_array` of finite nonnegative integer counts.

    Notes
    -----
    Network fallback is disabled and every feature type is retained; readers
    own feature selection. Scanpy owns source parsing, alignment, and source
    lifetime. Canonical CSR matrix storage can be wrapped
    as a sparse array without copying its buffers.
    Count checks inspect entries retained by Scanpy before local duplicate
    summation. Integer duplicates combine losslessly; decoded floating and
    boolean counts use a proved float32/float64 partial-sum bound before adding
    duplicates. An unproved floating bound raises a contextual precision error.
    Canonical decoded values retain their dtype. Source decoding can already
    round information; local reduction cannot recover it.
    """
    try:
        adata = _scanpy_read_10x_h5(path, gex_only=False, backup_url=None)
        return _normalize_counts(adata, path, context)
    except Exception as error:
        add_reader_context(error, context=context, path=path)
        raise


def read_10x_mtx(
    path: ArtifactDirectory,
    /,
    *,
    context: ReaderErrorContext,
    make_unique: bool = True,
    prefix: str | None = None,
    compressed: bool = True,
) -> AnnData:
    """Read one validated 10x Matrix Market family into owned sparse storage.

    Parameters
    ----------
    path
        Canonical directory previously validated by the shared path boundary.
        It contains the matrix, feature or gene, and barcode companion files.
    context
        Reader and artifact identity added to failures without changing their
        exception type or traceback.
    make_unique
        Whether Scanpy should make the gene-symbol ``var_names`` unique.
    prefix
        Filename prefix shared by the companion files, as discovered by the
        reader in ``path``. Scanpy concatenates it with the fixed companion
        basenames, so it must not contain a path separator.
    compressed
        Whether Scanpy should expect gzip-compressed v3 companion files.

    Returns
    -------
    anndata.AnnData
        An owned, eager, in-memory table whose ``X`` is a canonical
        :class:`scipy.sparse.csr_array` of finite nonnegative integer counts.

    Raises
    ------
    ReaderFormatError
        If the MatrixMarket header does not declare an integer coordinate
        matrix, or counts are negative, nonfinite or fractional.

    Notes
    -----
    Scanpy caching is disabled and every feature type is retained; readers own
    feature selection. Scanpy owns source parsing, alignment, and source
    lifetime. Only the MatrixMarket header is inspected outside Scanpy. Its parser can
    truncate malformed integer tokens and sum duplicates before returning;
    validation of its result cannot recover those source records. Retained
    duplicates use checked integer or bound-proven floating arithmetic.
    Canonical decoded float32 values retain their upstream precision limits.
    """
    try:
        _validate_mtx_header(path, context=context, prefix=prefix, compressed=compressed)
        adata = _scanpy_read_10x_mtx(
            path,
            var_names="gene_symbols",
            make_unique=make_unique,
            cache=False,
            gex_only=False,
            prefix=prefix,
            compressed=compressed,
        )
        return _normalize_counts(adata, path, context)
    except Exception as error:
        add_reader_context(error, context=context, path=path)
        raise
