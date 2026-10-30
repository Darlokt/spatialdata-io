"""Typed public-Scanpy boundaries for 10x count containers."""

from __future__ import annotations

import gzip
from typing import TYPE_CHECKING, Literal

import numpy as np
from scanpy import read_10x_h5 as _scanpy_read_10x_h5
from scanpy import read_10x_mtx as _scanpy_read_10x_mtx
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.table import normalize_owned_anndata

if TYPE_CHECKING:
    from pathlib import Path

    from anndata import AnnData

    from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile

_VALIDATION_BATCH_SIZE = 1_048_576

_WINDOWS_RESERVED_FILENAME_CHARACTERS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"COM{suffix}" for suffix in "123456789¹²³"}
    | {f"LPT{suffix}" for suffix in "123456789¹²³"}
)


def _normalize_counts(adata: AnnData, path: Path, context: ReaderErrorContext) -> AnnData:
    # The shared owner rejects these containers before even accessing X.
    if adata.is_view or adata.isbacked:
        return normalize_owned_anndata(adata)

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
    values = matrix.data
    if values.size and values.min() < 0:
        raise ReaderFormatError(
            context=context, path=path, found="negative counts", expected="nonnegative count values"
        )
    if values.dtype.kind == "f":
        for start in range(0, values.size, _VALIDATION_BATCH_SIZE):
            batch = values[start : start + _VALIDATION_BATCH_SIZE]
            # NaN and infinities remain the shared normalizer's responsibility.
            if np.any(batch > np.floor(batch)):
                raise ReaderFormatError(
                    context=context, path=path, found="fractional counts", expected="integer count values"
                )
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


def read_10x_h5(
    path: ArtifactFile,
    /,
    *,
    context: ReaderErrorContext,
    genome: str | None = None,
    gex_only: bool = True,
) -> AnnData:
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
    genome
        Optional genome forwarded to :func:`scanpy.read_10x_h5`.
    gex_only
        Whether Scanpy should retain only ``Gene Expression`` features.

    Returns
    -------
    anndata.AnnData
        An owned, eager, in-memory table whose ``X`` is a canonical
        :class:`scipy.sparse.csr_array` of finite nonnegative integer counts.

    Notes
    -----
    Network fallback is disabled. Scanpy owns source parsing, filtering,
    alignment, and source lifetime. Canonical CSR matrix storage can be wrapped
    as a sparse array without copying its buffers.
    Count checks inspect entries retained by Scanpy before local duplicate
    summation. Filtering can discard unselected source entries. Float32 storage
    is retained; source integers above its exact range can already be rounded.
    """
    try:
        adata = _scanpy_read_10x_h5(
            path,
            genome=genome,
            gex_only=gex_only,
            backup_url=None,
        )
        return _normalize_counts(adata, path, context)
    except Exception as error:
        add_reader_context(error, context=context, path=path)
        raise


def _validate_mtx_prefix(prefix: str | None) -> None:
    if prefix is None or prefix == "":
        return

    has_reserved_character = any(
        character < " " or character in _WINDOWS_RESERVED_FILENAME_CHARACTERS for character in prefix
    )
    companion_stem = f"{prefix}matrix.mtx".partition(".")[0].rstrip(" ").upper()
    if has_reserved_character or companion_stem in _WINDOWS_RESERVED_DEVICE_NAMES:
        message = (
            "prefix must be portable filename text without control characters, path separators, "
            "Windows-reserved characters, drives, or device names."
        )
        raise ValueError(message)


# Scanpy's independent source options stay explicit; an options bag or kwargs
# would erase this boundary's static contract.
def read_10x_mtx(  # noqa: PLR0913, RUF100
    path: ArtifactDirectory,
    /,
    *,
    context: ReaderErrorContext,
    var_names: Literal["gene_symbols", "gene_ids"] = "gene_symbols",
    make_unique: bool = True,
    gex_only: bool = True,
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
    var_names
        Feature field Scanpy should use for ``var_names``.
    make_unique
        Whether Scanpy should make selected variable names unique.
    gex_only
        Whether Scanpy should retain only ``Gene Expression`` features.
    prefix
        Portable lexical text prepended to every companion filename. Control
        characters, path separators, Windows-reserved characters, drives, and
        generated device names are rejected before source access.
    compressed
        Whether Scanpy should expect gzip-compressed v3 companion files.

    Returns
    -------
    anndata.AnnData
        An owned, eager, in-memory table whose ``X`` is a canonical
        :class:`scipy.sparse.csr_array` of finite nonnegative integer counts.

    Raises
    ------
    ValueError
        If ``prefix`` is not portable filename text or can alter the validated
        source directory.

    Notes
    -----
    Scanpy caching is disabled. Scanpy owns source parsing, filtering,
    alignment, and source lifetime. Literal ``.`` and ``..`` prefixes are safe
    because Scanpy concatenates them with fixed companion basenames.
    Only the MatrixMarket header is inspected outside Scanpy. Its parser can
    truncate malformed integer tokens and sum duplicates before returning;
    validation of its result cannot recover those source records. Float32
    storage is retained, including its integer precision limits.
    """
    try:
        _validate_mtx_prefix(prefix)
        _validate_mtx_header(path, context=context, prefix=prefix, compressed=compressed)
        adata = _scanpy_read_10x_mtx(
            path,
            var_names=var_names,
            make_unique=make_unique,
            cache=False,
            gex_only=gex_only,
            prefix=prefix,
            compressed=compressed,
        )
        return _normalize_counts(adata, path, context)
    except Exception as error:
        add_reader_context(error, context=context, path=path)
        raise
