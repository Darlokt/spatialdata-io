"""Bounded primitive square-GEF reads and exact sparse construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_array

from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers._utils.table import (
    as_csr_array,
    count_accumulation_dtype,
    finalize_count_arrays,
    normalize_sparse_duplicates,
)
from spatialdata_io.readers.stereoseq import _schemas

if TYPE_CHECKING:
    import numpy.typing as npt
    from scipy.sparse import csr_array

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.stereoseq._layout import SquareSource

_EXPRESSION_BUFFER_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SquareCounts:
    """Own one stored-bin table and its normalized-grid coordinates."""

    table: ad.AnnData
    coordinates: npt.NDArray[np.int64]
    bin_size: int
    origin: tuple[int, int]
    resolution: int
    sample_id: str | None


def _aggregate_counts(
    values: npt.NDArray[np.unsignedinteger],
    observation_indices: npt.NDArray[np.int64],
    feature_indptr: npt.NDArray[np.int64],
    observations: int,
    path: ArtifactFile,
) -> csr_array:
    """Use floating addition only when every nonnegative partial sum is exact."""
    # Native GEF records are grouped by gene, so CSC directly expresses their
    # spans without allocating a feature index for every expression record.
    maximum_records = int(np.diff(feature_indptr).max(initial=0))
    bound = int(values.max(initial=0)) * maximum_records
    shape = (observations, len(feature_indptr) - 1)
    dtype = count_accumulation_dtype(bound)
    if dtype is not None:
        return as_csr_array(
            csc_array((values.astype(dtype, copy=False), observation_indices, feature_indptr), shape=shape)
        )
    # Exceptional wide inputs retain exact integers until completed coordinate
    # sums are known. The shared fallback never rounds source integers first.
    try:
        return normalize_sparse_duplicates(csc_array((values, observation_indices, feature_indptr), shape=shape))
    except ValueError as error:
        raise _schemas.format_error(path, "square count aggregation", str(error), "lossless completed sums") from error


def _coordinate_rows(coordinates: npt.NDArray[np.int64]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Assign first-occurrence IDs with a lossless fixed-width coordinate key."""
    # Each source axis was validated as one int32 or uint32 dtype. Reducing
    # that axis modulo 2**32 is therefore injective, including negative int32
    # and uint32 boundary coordinates. Return the original int64 coordinates.
    keys = coordinates[:, 0].astype(np.uint64)
    keys <<= 32
    keys |= coordinates[:, 1].astype(np.uint32)
    _, first_occurrence_indices, inverse_rows = np.unique(keys, return_index=True, return_inverse=True)
    first_occurrence_order = np.argsort(first_occurrence_indices)
    row_ids_by_unique_key = np.empty(len(first_occurrence_indices), dtype=np.int64)
    row_ids_by_unique_key[first_occurrence_order] = np.arange(len(first_occurrence_indices), dtype=np.int64)
    return coordinates[first_occurrence_indices[first_occurrence_order]], row_ids_by_unique_key[inverse_rows]


def _read_records(
    expression: h5py.Dataset,
    exons: h5py.Dataset | None,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.unsignedinteger],
    npt.NDArray[np.unsignedinteger] | None,
]:
    """Read square records in byte-bounded slices into numerical buffers."""
    coordinates = np.empty((len(expression), 2), dtype=np.int64)
    counts = np.empty(len(expression), dtype=expression.dtype["count"])
    exon_counts = None if exons is None else np.empty(len(expression), dtype=exons.dtype)

    record_bytes = expression.dtype.itemsize + (0 if exons is None else exons.dtype.itemsize)
    records_per_slice = max(1, _EXPRESSION_BUFFER_BYTES // record_bytes)
    for start in range(0, len(expression), records_per_slice):
        stop = min(start + records_per_slice, len(expression))
        block = expression[start:stop]
        coordinates[start:stop, 0] = block["x"]
        coordinates[start:stop, 1] = block["y"]
        counts[start:stop] = block["count"]

        if exons is not None and exon_counts is not None:
            exon_counts[start:stop] = exons[start:stop]

    return coordinates, counts, exon_counts


def _read_bin(group: h5py.Group, bin_size: int, path: ArtifactFile, sample_id: str | None) -> SquareCounts:
    genes = _schemas.require_dataset(group, "gene", path)
    expression = _schemas.require_dataset(group, "expression", path)
    schema = _schemas.feature_schema(genes, path, kind="square")
    _schemas.expression_schema(expression, path, kind="square")

    # h5py returns unparameterized arrays; the validated compound dtype supplies
    # the named structured-array contract used below.
    features = cast("npt.NDArray[np.void]", genes[:])
    indptr = _schemas.validate_spans(features["offset"], features[schema.count], len(expression), path, genes.name)
    annotations = _schemas.feature_annotations(features, schema, path)
    exons = _schemas.optional_counts(group, "exon", len(expression), path)

    origin = (
        _schemas.integer_attribute(expression.attrs, "minX", path, expression.name),
        _schemas.integer_attribute(expression.attrs, "minY", path, expression.name),
    )
    resolution = _schemas.integer_attribute(expression.attrs, "resolution", path, expression.name, positive=True)
    coordinates, counts, exon_counts = _read_records(expression, exons)

    # Bin1 can retain repeated coordinate-feature events; coarser producer bins
    # sum them. Promote first so both paths have the same additive semantics:
    # https://github.com/STOmics/geftools/blob/91998d84dd7a2f311304aca7f44122512bce8eae/bin_task.cpp#L11-L76
    coordinates, observation_indices = _coordinate_rows(coordinates)
    matrix = _aggregate_counts(counts, observation_indices, indptr, len(coordinates), path)
    matrices = (matrix,)
    if exon_counts is not None:
        matrices += (_aggregate_counts(exon_counts, observation_indices, indptr, len(coordinates), path),)
    completed = finalize_count_arrays(
        matrices, context=ReaderErrorContext("stereoseq", f"square bin{bin_size} counts"), path=path
    )
    table = ad.AnnData(
        completed[0], obs=pd.DataFrame(index=pd.Index([str(row) for row in range(len(coordinates))])), var=annotations
    )
    if exon_counts is not None:
        table.layers["exon"] = completed[1]

    table.obsm["spatial"] = coordinates
    table.uns["stereoseq"] = {"origin": list(origin), "resolution": resolution}
    return SquareCounts(table, coordinates, bin_size, origin, resolution, sample_id)


def read_square(source: SquareSource) -> tuple[SquareCounts, ...]:
    """Read only selected stored bins under one HDF5 resource owner."""
    try:
        with h5py.File(source.file, "r") as handle:
            _schemas.validate_transcriptomics(handle.attrs, source.file)
            group = _schemas.require_group(handle, "geneExp", source.file)

            available: dict[int, str] = {}
            for name in group:
                if not name.startswith("bin") or not name[3:].isdigit() or int(name[3:]) <= 0:
                    continue
                size = int(name[3:])
                if size in available:
                    raise _schemas.format_error(source.file, "geneExp", repr(list(group)), "unique numeric bin names")
                available[size] = name

            selected = tuple(sorted(available)) if source.bins == "all" else source.bins
            if not selected or any(size not in available for size in selected):
                raise _schemas.format_error(
                    source.file, "geneExp", f"available bins {sorted(available)}", f"selected stored bins {selected}"
                )
            sample = _schemas.sample_id(handle, source.file)

            return tuple(
                _read_bin(_schemas.require_group(group, available[size], source.file), size, source.file, sample)
                for size in selected
            )
    except OSError as error:
        add_reader_context(error, context=ReaderErrorContext("stereoseq", "square counts"), path=source.file)
        raise
