"""Typed public-Scanpy boundaries for 10x count containers.

Scanpy parses the 10x HDF5 and Matrix Market families; this module validates
its result once and returns finite nonnegative integer counts as one canonical
float32 or float64 CSR array. Scanpy decodes int32 HDF5 values to float32, and
its Matrix Market parser already sums repeated records. The HDF5 format
stores at most one entry per (barcode, feature), so a repeated entry is
rejected rather than summed.
"""

from __future__ import annotations

import gzip
from typing import TYPE_CHECKING, cast

from scanpy import read_10x_h5 as _scanpy_read_10x_h5
from scanpy import read_10x_mtx as _scanpy_read_10x_mtx

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.table import finalize_count_arrays, normalize_owned_anndata

if TYPE_CHECKING:
    from pathlib import Path

    from anndata import AnnData
    from scipy.sparse import csr_array

    from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile
    from spatialdata_io.readers._utils.table import SparseDuplicateSummary


def _finalize_counts(adata: AnnData, path: Path, context: ReaderErrorContext) -> AnnData:
    """Validate the Scanpy count matrix once and store it as canonical float32 or float64 CSR."""
    if adata.X is None:
        raise ReaderFormatError(context=context, path=path, found="missing count data", expected="a count matrix")
    duplicates: list[SparseDuplicateSummary] = []
    try:
        normalize_owned_anndata(adata, duplicate_sums=duplicates)
    except ValueError as error:
        raise ReaderFormatError(
            context=context, path=path, found=str(error), expected="a well-formed sparse matrix of finite counts"
        ) from error
    if duplicates:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"{duplicates[0].combined_entries} repeated (barcode, feature) entries",
            expected="at most one stored count per barcode and feature",
        )
    adata.X = finalize_count_arrays((cast("csr_array", adata.X),), context=context, path=path)[0]
    return adata


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
        :class:`scipy.sparse.csr_array` of finite nonnegative integer counts:
        float32 when every count is at most 2**24 or Scanpy decoded float32,
        otherwise float64.

    Raises
    ------
    ReaderFormatError
        If the count matrix is missing or malformed, stores more than one
        entry for a (barcode, feature) pair, or holds nonfinite, negative,
        fractional or inexactly representable counts.

    Notes
    -----
    Network fallback is disabled and every feature type is retained; readers
    own feature selection. Scanpy owns source parsing, alignment, and source
    lifetime. Scanpy decodes int32 values to float32 before any local check,
    and local validation cannot recover information lost by that decoding.
    Canonical float32 storage is reused without copying its buffers.
    """
    try:
        adata = _scanpy_read_10x_h5(path, gex_only=False, backup_url=None)
        return _finalize_counts(adata, path, context)
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
        :class:`scipy.sparse.csr_array` of finite nonnegative integer counts:
        float32 when every count is at most 2**24 or Scanpy decoded float32,
        otherwise float64.

    Raises
    ------
    ReaderFormatError
        If the MatrixMarket header does not declare an integer coordinate
        matrix, or the parsed counts are missing, malformed, negative or
        inexactly representable.

    Notes
    -----
    Scanpy caching is disabled and every feature type is retained; readers own
    feature selection. Scanpy owns source parsing, alignment, and source
    lifetime. Only the MatrixMarket header is inspected outside Scanpy. Its
    parser can truncate malformed integer tokens and sums repeated records
    before returning; validation of its result cannot recover those source
    records.
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
        return _finalize_counts(adata, path, context)
    except Exception as error:
        add_reader_context(error, context=context, path=path)
        raise
