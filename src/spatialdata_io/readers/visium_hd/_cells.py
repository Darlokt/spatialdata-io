"""Read Space Ranger 4 cell matrices and aggregate raw 2 µm bins into nucleus counts.

Segmentation writes one cell-feature matrix per population::

    segmented_outputs/
    ├── filtered_feature_cell_matrix.h5    # cells passing Space Ranger's filter
    └── raw_feature_cell_matrix.h5         # every segmented cell

Cell barcodes are ``cellid_<id:09d>-1``; the positive integer ``id`` equals the
GeoJSON ``cell_id`` of the cell and nucleus polygons and becomes the table's
``instance_id`` as uint32.

10x writes no nucleus matrix. A nucleus table sums the raw 2 µm counts
(``binned_outputs/square_002um/raw_feature_bc_matrix.h5``) over every 2 µm
square that ``barcode_mappings.parquet`` marks ``in_nucleus`` for a cell,
including squares outside the tissue mask, in the order and population of the
selected cell matrix. Every raw 2 µm barcode must be mapped exactly once, and
the raw and cell matrices must list the same features.

Sums run in float32 when a Python-integer bound (largest decoded count times
the most squares contributing to one nucleus) is at most 2**24, otherwise in
float64 when at most 2**53, so every partial sum of nonnegative integers is
exact; a larger bound raises. The completed counts are stored as float32 when
every nucleus count is at most 2**24, otherwise as float64. Counts stay sparse
throughout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_array

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.table import as_csr_array, count_accumulation_dtype, finalize_count_arrays
from spatialdata_io.readers.visium_hd._schemas import MAPPINGS_SCHEMA, validate_mappings, validate_schema
from spatialdata_io.readers.visium_hd._tables import read_counts

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.paths import ArtifactFile

_BATCH_SIZE = 65_536


def cell_ids(names: pd.Index[str], path: ArtifactFile, *, version: str | None) -> NDArray[np.uint32]:
    """Decode canonical positive cell barcodes without accepting alternate spellings."""
    context = ReaderErrorContext("visium_hd", "cell barcodes", version)
    valid = names.str.fullmatch(r"cellid_[0-9]{9}-1", na=False)
    if not valid.all():
        raise ReaderFormatError(
            context=context,
            path=path,
            found="noncanonical cell barcode",
            expected="cellid_{positive_id:09d}-1",
        )

    values = names.str.slice(7, 16).to_numpy(dtype=np.uint32)
    if np.any(values == 0):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="zero cell ID",
            expected="positive cell IDs",
        )

    return values


def read_cell_table(path: ArtifactFile, *, make_var_names_unique: bool, version: str | None) -> ad.AnnData:
    """Preserve the chosen raw or filtered cell population and source order."""
    table = read_counts(path, version=version)
    table.obs["instance_id"] = cell_ids(table.obs_names, path, version=version)
    if make_var_names_unique:
        table.var_names_make_unique()

    return table


def _aggregate_counts(  # noqa: PLR0913, RUF100 - Explicit operands, target size, source and version.
    counts: csr_array,
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    observations: int,
    path: ArtifactFile,
    *,
    version: str | None,
) -> csr_array:
    """Choose floating arithmetic that preserves every nonnegative partial sum."""
    if not rows.size:
        return csr_array((observations, counts.shape[1]), dtype=np.float32)

    contributing = np.bincount(rows, minlength=observations)
    bound = int(counts.data.max(initial=0)) * int(contributing.max(initial=0))
    dtype = count_accumulation_dtype(bound)
    if dtype is None:
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "nucleus aggregation", version),
            path=path,
            found="float64 exact-sum bound exceeds 2**53",
            expected="maximum decoded count times maximum contributing bins at most 2**53",
        )

    # Let SciPy widen values during multiplication. Only narrow a source that
    # would otherwise force wider arithmetic, sharing its immutable structure.
    operand = counts
    if np.result_type(counts.dtype, dtype) != dtype:
        operand = csr_array((counts.data.astype(dtype), counts.indices, counts.indptr), shape=counts.shape, copy=False)
    operator = csr_array((np.ones(rows.size, dtype=dtype), (rows, columns)), shape=(observations, counts.shape[0]))
    aggregated = operator @ operand
    # SciPy sparse products need not have sorted indices, even with canonical inputs.
    return as_csr_array(aggregated)


def read_nucleus_table(
    raw_file: ArtifactFile,
    mapping_file: ArtifactFile,
    population: ad.AnnData,
    known_cell_ids: pd.Index,
    *,
    version: str | None,
) -> ad.AnnData:
    """Sum raw 2 µm bins, including out-of-tissue bins, in selected cell order."""
    context = ReaderErrorContext("visium_hd", "nucleus aggregation", version)
    mapping_context = ReaderErrorContext("visium_hd", "barcode mappings", version)
    raw = read_counts(raw_file, version=version)
    if any(
        not np.array_equal(raw.var[column].to_numpy(), population.var[column].to_numpy())
        for column in ("gene_ids", "genome", "feature_types")
    ):
        raise ReaderFormatError(
            context=context,
            path=raw_file,
            found="different feature IDs/order",
            expected="the same features in raw bins and cell counts",
        )

    # read_counts guarantees finite nonnegative integral decoded CSR values.
    imported = cast("csr_array", raw.X)
    source_rows: list[NDArray[np.int64]] = []
    target_rows: list[NDArray[np.int64]] = []
    seen = np.zeros(raw.n_obs, dtype=np.bool_)
    # Use pandas' Arrow-backed string index engine on both sides of alignment;
    # mixed string backends otherwise recast the entire count index per batch.
    barcode_dtype = pd.StringDtype(storage="pyarrow", na_value=np.nan)
    raw_names = raw.obs_names.astype(barcode_dtype, copy=False)
    population_names = population.obs_names.astype(barcode_dtype, copy=False)

    with pq.ParquetFile(mapping_file) as source:
        validate_schema(source.schema_arrow, MAPPINGS_SCHEMA, mapping_file, context=mapping_context)
        for batch in source.iter_batches(batch_size=_BATCH_SIZE, columns=MAPPINGS_SCHEMA.names):
            validate_mappings(batch, mapping_file, context=mapping_context)
            barcodes: pd.Index[str] = pd.Index(batch.column("square_002um"), dtype=barcode_dtype)
            indices = raw_names.get_indexer(barcodes)
            if np.any(indices < 0) or np.unique(indices).size != indices.size or np.any(seen[indices]):
                raise ReaderFormatError(
                    context=context,
                    path=mapping_file,
                    found="unknown or repeated 2 µm barcode",
                    expected="one mapping per raw 2 µm bin",
                )
            seen[indices] = True

            members = batch.column("in_nucleus").to_numpy(zero_copy_only=False)
            assigned: pd.Index[str] = pd.Index(batch.column("cell_id"), dtype=barcode_dtype)
            nonnull = assigned.dropna().unique()
            if not pd.Index(cell_ids(nonnull, mapping_file, version=version)).isin(known_cell_ids).all():
                raise ReaderFormatError(
                    context=context,
                    path=mapping_file,
                    found="unknown mapped cell ID",
                    expected="all assigned cells in nucleus geometry",
                )

            # Valid cells outside the selected raw/filtered population do not contribute.
            cells = population_names.get_indexer(assigned)
            keep = members & (cells >= 0)
            source_rows.append(indices[keep])
            target_rows.append(cells[keep])

    if not seen.all():
        raise ReaderFormatError(
            context=context,
            path=mapping_file,
            found="incomplete barcode mappings",
            expected="one mapping per raw 2 µm bin",
        )

    rows = np.concatenate(target_rows) if target_rows else np.empty(0, dtype=np.int64)
    columns = np.concatenate(source_rows) if source_rows else np.empty(0, dtype=np.int64)
    aggregated = _aggregate_counts(imported, rows, columns, population.n_obs, raw_file, version=version)
    completed = finalize_count_arrays((aggregated,), context=context, path=raw_file)[0]
    # Both annotations are eager frames from the shared count boundary, never Dataset2D.
    return ad.AnnData(
        completed,
        obs=cast("pd.DataFrame", population.obs).copy(),
        var=cast("pd.DataFrame", population.var).copy(),
    )
