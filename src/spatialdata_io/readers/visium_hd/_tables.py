"""Exact count-source validation and final linkage for Visium HD tables."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import h5py
import numpy as np
import pandas as pd

from spatialdata_io.readers._utils.counts import read_10x_h5
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table, prepare_anndata_dataframe

if TYPE_CHECKING:
    from anndata import AnnData

    from spatialdata_io.readers._utils.paths import ArtifactFile

# Scanpy returns float32. Validate source integers before its conversion can round them.
MAX_EXACT_COUNT = 2**24
_COUNT_BATCH_SIZE = 1_048_576
_MATRIX_DIMENSIONS = 2


def _count_error(path: ArtifactFile, *, found: str, expected: str) -> ReaderFormatError:
    return ReaderFormatError(
        context=ReaderErrorContext("visium_hd", "count matrix"), path=path, found=found, expected=expected
    )


def _source_arrays(handle: h5py.File, path: ArtifactFile) -> tuple[h5py.Dataset, h5py.Dataset, h5py.Dataset, int]:
    """Validate the CSC structural arrays needed to prove canonical exact counts."""
    try:
        arrays = tuple(handle[f"matrix/{name}"] for name in ("data", "indices", "indptr", "shape"))
    except KeyError as error:
        raise _count_error(
            path, found="missing sparse matrix dataset", expected="matrix/data, indices, indptr and shape"
        ) from error

    for array in arrays:
        if not isinstance(array, h5py.Dataset) or array.ndim != 1 or array.dtype.kind not in "iu":
            raise _count_error(
                path,
                found="noninteger or nonvector sparse matrix dataset",
                expected="one-dimensional integer data, indices, indptr and shape datasets",
            )

    # The loop proves all four HDF5 objects are datasets; h5py stubs do not preserve that narrowing.
    data, indices, pointers, shape = cast("tuple[h5py.Dataset, h5py.Dataset, h5py.Dataset, h5py.Dataset]", arrays)
    if len(shape) != _MATRIX_DIMENSIONS:
        raise _count_error(path, found="invalid matrix shape", expected="two positive dimensions (features, barcodes)")

    dimensions = shape[:]
    if np.any(dimensions <= 0) or np.any(dimensions > np.iinfo(np.int64).max):
        raise _count_error(path, found="invalid matrix dimensions", expected="positive int64 dimensions")

    features, observations = int(dimensions[0]), int(dimensions[1])
    if len(data) != len(indices) or len(pointers) != observations + 1:
        raise _count_error(
            path,
            found="inconsistent sparse matrix lengths",
            expected="one index per count and one pointer per barcode plus one",
        )
    if int(pointers[0]) != 0 or int(pointers[-1]) != len(data):
        raise _count_error(
            path, found="invalid sparse pointer endpoints", expected="indptr starts at zero and ends at nnz"
        )

    return data, indices, pointers, features


def _validate_counts(handle: h5py.File, path: ArtifactFile) -> None:
    """Scan source CSC once, bounding pointer, index and count decoding buffers."""
    data, indices, pointers, features = _source_arrays(handle, path)
    for column in range(0, len(pointers) - 1, _COUNT_BATCH_SIZE):
        # Adjacent pointer batches share an endpoint, so no column boundary is skipped.
        stops = pointers[column : column + _COUNT_BATCH_SIZE + 1]
        if np.any(stops > len(data)) or np.any(stops < 0) or np.any(stops[1:] < stops[:-1]):
            raise _count_error(
                path, found="invalid sparse pointers", expected="nondecreasing indptr values in [0, nnz]"
            )

        previous: int | None = None
        for start in range(int(stops[0]), int(stops[-1]), _COUNT_BATCH_SIZE):
            end = min(start + _COUNT_BATCH_SIZE, int(stops[-1]))
            entries = indices[start:end]
            values = data[start:end]
            if np.any(entries < 0) or np.any(entries >= features):
                raise _count_error(
                    path, found="out-of-range sparse feature index", expected="indices within the feature dimension"
                )

            # Feature indices may restart only at a barcode boundary in the CSC stream.
            repeated = np.flatnonzero(entries[1:] <= entries[:-1]) + start + 1
            if repeated.size and np.any(stops[np.searchsorted(stops, repeated)] != repeated):
                raise _count_error(
                    path,
                    found="duplicate or unsorted sparse coordinates",
                    expected="strictly increasing feature indices within every barcode",
                )
            if previous is not None and entries[0] <= previous and stops[np.searchsorted(stops, start)] != start:
                raise _count_error(
                    path,
                    found="duplicate or unsorted sparse coordinates across a batch",
                    expected="strictly increasing feature indices within every barcode",
                )
            previous = int(entries[-1])

            if np.any(values < 0) or np.any(values > MAX_EXACT_COUNT):
                raise _count_error(
                    path,
                    found="counts outside [0, 2**24]",
                    expected="nonnegative counts exactly representable by the supported float32 count reader",
                )


def read_counts(path: ArtifactFile) -> AnnData:
    """Read canonical 10x counts after proving integer-to-float32 exactness.

    Strictly increasing source feature indices prevent a float32 duplicate
    reduction from rounding a sum even when each stored source value is exact.
    """
    context = ReaderErrorContext("visium_hd", "count matrix")
    try:
        with h5py.File(path, "r") as handle:
            _validate_counts(handle, path)
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise

    table = read_10x_h5(path, context=context, gex_only=False)
    if not table.obs_names.is_unique:
        raise _count_error(path, found="duplicate barcodes", expected="unique barcodes")

    return table


def link_table(table: AnnData, target: str, identifiers: pd.Index) -> AnnData:
    """Link the final owned table once against its materialized target index."""
    context = ReaderErrorContext("visium_hd", target)
    table.obs["region"] = pd.Categorical([target] * table.n_obs, categories=pd.Index([target]))

    # Shared counts guarantee eager annotations; upstream types also include Dataset2D.
    prepare_anndata_dataframe(cast("pd.DataFrame", table.obs), context=context)
    prepare_anndata_dataframe(cast("pd.DataFrame", table.var), context=context)

    return parse_linked_table(
        table,
        TableLinkage((target,), "region", "instance_id"),
        context=context,
        expected_instance_ids={target: identifiers},
    )
