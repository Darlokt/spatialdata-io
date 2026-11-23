"""Read the stored CSC buffers and aligned annotations of standalone 10x HDF5 matrices."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from anndata import AnnData
from scipy.sparse import csr_array

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.table import finalize_count_arrays

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from pandas.api.extensions import ExtensionArray

    from spatialdata_io.readers._utils.paths import ArtifactFile


_TEXT_BATCH_SIZE = 262_144


def _local_item(parent: h5py.Group, key: str) -> h5py.Group | h5py.Dataset:
    if not isinstance(parent.get(key, getlink=True), h5py.HardLink):
        message = f"{parent.name}/{key} must be stored in this HDF5 file"
        raise TypeError(message)
    return parent[key]


def _group(parent: h5py.Group, key: str) -> h5py.Group:
    group = _local_item(parent, key)
    if not isinstance(group, h5py.Group):
        message = f"{group.name} must be an HDF5 group"
        raise TypeError(message)
    return group


def _vector(item: h5py.Group | h5py.Dataset, *, length: int | None = None) -> h5py.Dataset:
    if not isinstance(item, h5py.Dataset) or item.ndim != 1:
        message = f"{item.name} must be a one-dimensional HDF5 dataset"
        raise ValueError(message)
    if length is not None and len(item) != length:
        message = f"10x matrix shape requires {length} entries in {item.name}, found {len(item)}"
        raise ValueError(message)
    if item.is_virtual or item.external:
        message = f"{item.name} must store its data within the count file"
        raise ValueError(message)
    return item


def _text(dataset: h5py.Dataset) -> ExtensionArray:
    if h5py.check_string_dtype(dataset.dtype) is None:
        message = f"{dataset.name} must contain HDF5 strings"
        raise ValueError(message)
    # Keep only a batch of the original bytes and fixed-width padding alongside
    # the final UTF-8 buffers, even on slides with millions of barcodes.
    chunks = []
    for start in range(0, len(dataset), _TEXT_BATCH_SIZE):
        strings = pa.array(dataset[start : start + _TEXT_BATCH_SIZE], type=pa.large_string())
        if dataset.dtype.kind == "S":
            strings = pc.call_function("utf8_rtrim", [strings], options=pc.TrimOptions("\0"))
        chunks.append(strings)
    return pd.array(pa.chunked_array(chunks, type=pa.large_string()), dtype=pd.StringDtype(storage="pyarrow"))


def _matrix_group(handle: h5py.File) -> tuple[h5py.Group, bool]:
    if "matrix" in handle:
        return _group(handle, "matrix"), True
    if len(handle) != 1:
        message = "Legacy 10x HDF5 file is empty or contains more than one genome."
        raise ValueError(message)
    return _group(handle, next(iter(handle))), False


@contextmanager
def _matrix(path: ArtifactFile, context: ReaderErrorContext) -> Generator[tuple[h5py.Group, bool]]:
    try:
        # A sequential pass does not reuse HDF5 chunks between vectors.
        with h5py.File(path, "r", rdcc_nbytes=0) as handle:
            yield _matrix_group(handle)
    except ReaderFormatError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=str(error),
            expected="a well-formed 10x count matrix and aligned annotations",
        ) from error
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise


def _feature_group(matrix: h5py.Group) -> h5py.Group:
    if "features" not in matrix:
        message = "10x h5 has no features group"
        raise ValueError(message)
    return _group(matrix, "features")


def _string_index(group: h5py.Group, key: str, *, length: int | None = None) -> pd.Index[str]:
    return pd.Index(_text(_vector(_local_item(group, key), length=length)))


def read_10x_h5_axes(path: ArtifactFile, /, *, context: ReaderErrorContext) -> tuple[pd.Index[str], pd.Index[str]]:
    """Read the ordered feature names and barcodes without opening the expression datasets."""
    with _matrix(path, context) as (matrix, modern):
        names = _string_index(_feature_group(matrix), "name") if modern else _string_index(matrix, "gene_names")
        return names, _string_index(matrix, "barcodes")


def _features(matrix: h5py.Group, *, modern: bool, length: int) -> pd.DataFrame:
    if not modern:
        return pd.DataFrame(
            {"gene_ids": _text(_vector(_local_item(matrix, "genes"), length=length))},
            index=_string_index(matrix, "gene_names", length=length),
        )
    features = _feature_group(matrix)
    names = _string_index(features, "name", length=length)
    ids = _text(_vector(_local_item(features, "id"), length=length))
    fields: dict[str, ExtensionArray | np.ndarray] = (
        {"gene_ids": _text(_vector(_local_item(features, "gene_id"), length=length)), "probe_ids": ids}
        if "gene_id" in features
        else {"gene_ids": ids}
    )
    fields["feature_types"] = _text(_vector(_local_item(features, "feature_type"), length=length))
    for name in features:
        if name in {"id", "name", "gene_id", "feature_type", "_all_tag_keys"}:
            continue
        item = _local_item(features, name)
        if not isinstance(item, h5py.Dataset):
            continue
        if name in fields:
            message = f"feature annotation {name!r} conflicts with a 10x identifier column"
            raise ValueError(message)
        dataset = _vector(item, length=length)
        if h5py.check_string_dtype(dataset.dtype) is not None:
            fields[name] = _text(dataset)
        elif dataset.dtype.kind in "biuf":
            values = dataset[:]
            fields[name] = values if values.dtype.kind == "b" else values.astype(str)
        else:
            message = f"{dataset.name} must store text, boolean or numeric feature annotations"
            raise ValueError(message)
    return pd.DataFrame(fields, index=names, copy=False)


def _h5_counts(source: csr_array, path: Path, context: ReaderErrorContext) -> csr_array:
    """Sort owned, structurally validated CSR and reject repeated HDF5 entries."""
    source.sort_indices()
    if not source.has_canonical_format:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="repeated (barcode, feature) entries",
            expected="at most one stored count per barcode and feature",
        )
    return finalize_count_arrays((source,), context=context, path=path)[0]


def read_10x_h5_components(
    path: ArtifactFile, /, *, context: ReaderErrorContext
) -> tuple[csr_array, pd.DataFrame, pd.DataFrame]:
    """Read owned CSR counts, barcode annotations and feature annotations, in matrix order."""
    with _matrix(path, context) as (matrix, modern):
        shape = _vector(_local_item(matrix, "shape"))
        if shape.shape != (2,) or shape.dtype.kind not in "iu":
            message = "10x matrix shape must contain integer feature and barcode counts"
            raise ValueError(message)
        n_features, n_barcodes = map(int, shape[:])
        data, indices, pointers = (_vector(_local_item(matrix, key)) for key in ("data", "indices", "indptr"))
        if data.dtype.kind not in "biuf":
            message = "10x counts require a boolean, integer or real floating dtype."
            raise ValueError(message)
        if indices.dtype.kind not in "iu" or pointers.dtype.kind not in "iu":
            message = "10x matrix indices and pointers must be integers"
            raise ValueError(message)
        if len(pointers) != n_barcodes + 1 or len(indices) != len(data):
            message = "10x matrix sparse buffer lengths disagree with its shape"
            raise ValueError(message)
        var = _features(matrix, modern=modern, length=n_features)
        barcodes = _string_index(matrix, "barcodes", length=n_barcodes)
        flags = {}
        if "filtered_barcodes" in matrix:
            values = _vector(_local_item(matrix, "filtered_barcodes"), length=n_barcodes)[:]
            if values.dtype.kind not in "bui" or values.min(initial=0) < 0 or values.max(initial=0) > 1:
                message = "filtered_barcodes must contain boolean flags or integer zeros and ones"
                raise ValueError(message)
            flags["filtered_barcodes"] = values.astype(bool, copy=False)
        obs = pd.DataFrame(flags, index=barcodes, copy=False)
        indptr = pointers[:]
        if int(indptr[-1]) != len(data):
            message = "10x matrix pointers must span exactly the stored counts and indices"
            raise ValueError(message)
        if indptr[0] != 0 or np.any(indptr[1:] < indptr[:-1]):
            message = "10x matrix pointers must be nondecreasing from zero"
            raise ValueError(message)
        # Feature-by-barcode CSC is observation-by-feature CSR. HDF5 converts
        # byte order during the read, including for big-endian source values.
        feature_indices = indices[:]
        if feature_indices.size and (feature_indices.min() < 0 or feature_indices.max() >= n_features):
            message = "10x matrix contains a feature index outside its shape"
            raise ValueError(message)
        values = data.astype(data.dtype.newbyteorder("="))[:]
        expression = csr_array((values, feature_indices, indptr), shape=(len(obs), len(var)))
        return _h5_counts(expression, path, context), obs, var


def read_10x_h5(path: ArtifactFile, /, *, context: ReaderErrorContext) -> AnnData:
    """Read a modern feature/probe or single-genome legacy matrix without rounding integer counts."""
    counts, obs, var = read_10x_h5_components(path, context=context)
    return AnnData(counts, obs=obs, var=var)
