"""Native cell-GEF spans, exact expression and optional source borders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers._utils.table import as_csr_array
from spatialdata_io.readers.stereoseq import _schemas

if TYPE_CHECKING:
    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.stereoseq._layout import CellSource

_EXPRESSION_BUFFER_BYTES = 8 * 1024 * 1024
_BORDER_VERTICES = 32
_SPATIAL_DIMENSIONS = 2


@dataclass(frozen=True, slots=True)
class CellCounts:
    """Own native cell counts, source IDs and optional paired contours."""

    table: ad.AnnData
    coordinates: npt.NDArray[np.int64]
    cell_ids: npt.NDArray[np.uint64]
    borders: npt.NDArray[np.int16] | None
    origin: tuple[int, int]
    resolution: int
    sample_id: str | None


def _canonical_native(
    counts: npt.NDArray[np.uint64],
    feature_indices: npt.NDArray[np.int64],
    indptr: npt.NDArray[np.int64],
    features: int,
    path: ArtifactFile,
) -> csr_array:
    matrix = csr_array((counts, feature_indices, indptr), shape=(len(indptr) - 1, features))
    matrix.sort_indices()
    if not matrix.has_canonical_format:
        raise _schemas.format_error(
            path, "cellExp", "repeated feature within a cell", "unique native cell-feature records"
        )

    return as_csr_array(matrix)


def _read_expression(
    expression: h5py.Dataset,
    exons: h5py.Dataset | None,
    cell_indptr: npt.NDArray[np.int64],
    feature_count: int,
    path: ArtifactFile,
) -> tuple[csr_array, csr_array | None]:
    counts = np.empty(len(expression), dtype=np.uint64)
    feature_indices = np.empty(len(expression), dtype=np.int64)
    exon_values = None if exons is None else np.empty(len(expression), dtype=np.uint64)
    record_bytes = expression.dtype.itemsize + (0 if exons is None else exons.dtype.itemsize)
    records_per_slice = max(1, _EXPRESSION_BUFFER_BYTES // record_bytes)
    for start in range(0, len(expression), records_per_slice):
        stop = min(start + records_per_slice, len(expression))
        block = expression[start:stop]
        counts[start:stop] = block["count"]
        feature_indices[start:stop] = block["geneID"]
        if np.any(feature_indices[start:stop] >= feature_count):
            raise _schemas.format_error(
                path, expression.name, "out-of-range feature index", f"indices below {feature_count}"
            )
        if exons is not None and exon_values is not None:
            exon_values[start:stop] = exons[start:stop]

    # Each layer owns index buffers before sorting: counts and exons may arrive
    # unsorted, and mutating a shared index would break the other layer's values.
    exon_indices = None if exon_values is None else feature_indices.copy()
    matrix = _canonical_native(counts, feature_indices, cell_indptr, feature_count, path)
    exon_matrix = None
    if exon_values is not None and exon_indices is not None:
        exon_matrix = csr_array((exon_values, exon_indices, cell_indptr.copy()), shape=matrix.shape)
        exon_matrix.sort_indices()
        exon_matrix = as_csr_array(exon_matrix)
    return matrix, exon_matrix


def _borders(group: h5py.Group, cell_count: int, path: ArtifactFile) -> npt.NDArray[np.int16]:
    value = group.get("cellBorder")
    if (
        not isinstance(value, h5py.Dataset)
        or value.shape != (cell_count, _BORDER_VERTICES, _SPATIAL_DIMENSIONS)
        or value.dtype != np.dtype(np.int16)
    ):
        raise _schemas.format_error(path, "cellBorder", str(value), f"int16 shape ({cell_count}, 32, 2)")
    # The complete eager contour representation is required only in polygon mode.
    return cast("npt.NDArray[np.int16]", value[:])


def _read_cells(handle: h5py.File, source: CellSource, *, read_borders: bool) -> CellCounts:
    path = source.file
    group = _schemas.require_group(handle, "cellBin", path)
    cells_dataset = _schemas.require_dataset(group, "cell", path)
    columns = _schemas.cell_schema(cells_dataset, path)
    genes_dataset = _schemas.require_dataset(group, "gene", path)
    schema = _schemas.feature_schema(genes_dataset, path, kind="cell")
    expression = _schemas.require_dataset(group, "cellExp", path)
    _schemas.expression_schema(expression, path, kind="cell")
    exons = _schemas.optional_counts(group, "cellExpExon", len(expression), path)
    origin = (
        _schemas.integer_attribute(handle.attrs, "offsetX", path, "cell origin"),
        _schemas.integer_attribute(handle.attrs, "offsetY", path, "cell origin"),
    )
    resolution = _schemas.integer_attribute(handle.attrs, "resolution", path, "cell resolution", positive=True)
    sample = _schemas.sample_id(handle, path)

    # Named dtypes were validated before materializing required cell/feature metadata.
    cells = cast("npt.NDArray[np.void]", cells_dataset[:])
    genes = cast("npt.NDArray[np.void]", genes_dataset[:])
    cell_indptr = _schemas.validate_spans(
        cells["offset"], cells["geneCount"], len(expression), path, cells_dataset.name
    )
    ids = cells["id"].astype(np.uint64)
    if len(np.unique(ids)) != len(ids):
        raise _schemas.format_error(path, cells_dataset.name, "duplicate cell IDs", "unique source cell IDs")
    coordinates = np.column_stack((cells["x"], cells["y"])).astype(np.int64)

    observations = pd.DataFrame(index=pd.Index([str(identity) for identity in ids]))
    for name in columns:
        observations[name] = (
            pd.Categorical(cells[name]) if name in {"cellTypeID", "clusterID"} else _schemas.arrow_column(cells[name])
        )
    annotations = _schemas.feature_annotations(genes, schema, path)

    for name, frame in (("cellExon", observations), ("geneExon", annotations)):
        summary = _schemas.optional_counts(group, name, len(frame), path)
        if summary is not None:
            frame[name] = _schemas.arrow_column(summary[:])

    matrix, exon_matrix = _read_expression(expression, exons, cell_indptr, len(genes), path)
    table = ad.AnnData(matrix, obs=observations, var=annotations)
    if exon_matrix is not None:
        table.layers["exon"] = exon_matrix
    table.obsm["spatial"] = coordinates
    metadata: dict[str, int | list[int] | list[str]] = {"origin": list(origin), "resolution": resolution}
    table.uns["stereoseq"] = metadata

    if "cellTypeList" in group:
        names = _schemas.require_dataset(group, "cellTypeList", path)
        if names.dtype.kind != "S":
            raise _schemas.format_error(path, names.name, str(names.dtype), "fixed-width UTF-8 category text")
        metadata["cell_type_names"] = _schemas.decode_strings(names[:], path, names.name)

    borders = _borders(group, len(cells), path) if read_borders else None
    return CellCounts(table, coordinates, ids, borders, origin, resolution, sample)


def read_cells(source: CellSource, *, read_borders: bool) -> CellCounts:
    """Read one raw cell population with one visible HDF5 owner."""
    try:
        with h5py.File(source.file, "r") as handle:
            _schemas.validate_transcriptomics(handle.attrs, source.file)
            return _read_cells(handle, source, read_borders=read_borders)
    except OSError as error:
        add_reader_context(error, context=ReaderErrorContext("stereoseq", "cell counts"), path=source.file)
        raise
