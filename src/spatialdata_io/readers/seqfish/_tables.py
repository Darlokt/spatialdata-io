"""Read one ROI's count matrix and cell coordinates into an exact sparse AnnData table.

``<ROI>_CellxGene.csv`` holds one row per cell: the cell ID, then one count
per feature. ``<ROI>_CellCoordinates.csv`` holds the same cells, possibly in
another row order, with their area and centroid (see :mod:`._schemas`)::

    ,GeneB,GeneA                    label,area,center_x,center_y
    9,3,0                           2,12.57,4.5,1.25
    2,0,5                           9,3.14,1.5,0.5

The table keeps the count-row order and the feature-header order. Cell metadata
is aligned by exact ID, so both files must contain the same unique IDs. Arrow
reads the counts as typed record batches of about 1 MiB of text, each stored
as canonical CSR; the dense expression matrix is never allocated. ``area``
(µm²) becomes an observation column, ``(center_x, center_y)`` (µm) becomes
``obsm["spatial"]``, and the cell ID becomes ``obs["instance_id"]``.
Observation names are positional strings starting at ``"0"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from scipy import sparse

from spatialdata_io._positions import POSITIONS_KEY, PositionsMetadata
from spatialdata_io.readers._utils.csv import csv_errors, csv_header
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.table import (
    TableLinkage,
    as_csr_array,
    finalize_count_arrays,
    normalize_owned_anndata,
    parse_linked_table,
    prepare_anndata_dataframe,
)
from spatialdata_io.readers.seqfish._schemas import (
    CELL_TYPES,
    CELLS_CONTEXT,
    COUNTS_CONTEXT,
    count_types,
    validate_cell_columns,
    validate_cell_ids,
)

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile

_TABLE_CONTEXT = ReaderErrorContext("seqfish", "table")
# Source text per count batch. A count takes at least two bytes of text and
# eight as uint64, so a batch holds at most 4 MiB of values plus their dense copy.
_BATCH_BYTES = 1024 * 1024
# Quoted fields may contain newlines.
_PARSE = pacsv.ParseOptions(newlines_in_values=True)


def _count_block(batch: pa.RecordBatch) -> sparse.csr_array:
    """Convert the count columns of one typed batch to canonical CSR."""
    counts = batch.select(range(1, batch.num_columns)).to_tensor(row_major=True)
    return as_csr_array(np.asarray(counts))


def _read_counts(path: ArtifactFile) -> tuple[sparse.csr_array, pd.Index, tuple[str, ...]]:
    """Accumulate canonical sparse blocks from bounded typed batches."""
    types = count_types(csv_header(path, context=COUNTS_CONTEXT), path)
    ids: list[pa.Array] = []
    blocks: list[sparse.csr_array] = []
    with (
        csv_errors(context=COUNTS_CONTEXT, path=path),
        pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(block_size=_BATCH_BYTES, use_threads=False),
            parse_options=_PARSE,
            convert_options=pacsv.ConvertOptions(
                column_types=types, include_columns=types.names, null_values=[], strings_can_be_null=False
            ),
        ) as reader,
    ):
        for batch in reader:
            ids.append(batch.column(0))
            # Rows are never summed, so a batch's counts are complete once read.
            # Finalizing it now retains float32 values instead of 8-byte
            # integers; stacking promotes every block exactly to float64 if any
            # block needs it, which is the dtype one final call would select.
            blocks.extend(finalize_count_arrays((_count_block(batch),), context=COUNTS_CONTEXT, path=path))

    if not blocks:
        raise ReaderFormatError(
            context=COUNTS_CONTEXT, path=path, found="no count rows", expected="a nonempty cell population"
        )
    identifiers = validate_cell_ids(pa.chunked_array(ids, types.field(0).type), context=COUNTS_CONTEXT, path=path)
    return sparse.vstack(blocks, format="csr"), identifiers, tuple(types.names[1:])


def _read_cells(path: ArtifactFile) -> pd.DataFrame:
    """Project cell metadata and validate its independent identity population."""
    validate_cell_columns(csv_header(path, context=CELLS_CONTEXT), path)
    with csv_errors(context=CELLS_CONTEXT, path=path):
        source = pacsv.read_csv(
            path,
            parse_options=_PARSE,
            convert_options=pacsv.ConvertOptions(
                column_types=CELL_TYPES, include_columns=list(CELL_TYPES), null_values=[], strings_can_be_null=False
            ),
        )
    for name in ("area", "center_x", "center_y"):
        if not np.isfinite(source[name].to_numpy()).all():
            raise ReaderFormatError(
                context=CELLS_CONTEXT, path=path, found=f"nonfinite {name!r} values", expected=f"finite {name!r} values"
            )
    if (source["area"].to_numpy() < 0).any():
        raise ReaderFormatError(
            context=CELLS_CONTEXT, path=path, found="negative cell area", expected="nonnegative cell areas"
        )
    ids = validate_cell_ids(source["label"], context=CELLS_CONTEXT, path=path)
    return source.drop_columns(["label"]).to_pandas(types_mapper=pd.ArrowDtype).set_axis(ids)


def read_table(counts: ArtifactFile, cells: ArtifactFile) -> ad.AnnData:
    """Return an owned unlinked CSR table in count-row and feature-header order."""
    expression, identifiers, features = _read_counts(counts)
    metadata = _read_cells(cells)
    if not metadata.index.equals(identifiers):
        positions = metadata.index.get_indexer(identifiers)
        if len(metadata) != len(identifiers) or (positions < 0).any():
            raise ReaderFormatError(
                context=CELLS_CONTEXT,
                path=cells,
                found="different count and metadata IDs",
                expected="identical unique cell populations",
            )
        metadata = metadata.iloc[positions]

    observations = pd.DataFrame(index=pd.Index([str(i) for i in range(len(identifiers))]))
    observations["area"] = pd.Series(metadata["area"].array, index=observations.index)
    observations["instance_id"] = pd.Series(identifiers.array, index=observations.index)
    table = ad.AnnData(X=expression, obs=observations, var=pd.DataFrame(index=pd.Index(features)))
    table.obsm["spatial"] = metadata[["center_x", "center_y"]].to_numpy(dtype=np.float64)
    return table


def select_cells(table: ad.AnnData, instance_ids: pd.Index, *, counts: ArtifactFile) -> ad.AnnData:
    """Return an owned table of the cells in ``instance_ids``, keeping row names and order.

    The retained counts select their dtype again, so dropping the only cell
    that needed float64 gives the float32 table the remaining cells would
    have produced on their own.
    """
    # The reader constructs an eager owned AnnData, never a backed Dataset2D.
    observations = cast("pd.DataFrame", table.obs)
    subset = table[observations["instance_id"].isin(instance_ids).to_numpy()].copy()
    (subset.X,) = finalize_count_arrays((cast("sparse.csr_array", subset.X),), context=COUNTS_CONTEXT, path=counts)
    return subset


def finalize_table(table: ad.AnnData, *, frame: str, region: str | None, counts: ArtifactFile) -> ad.AnnData:
    """Finalize owned annotations once and link only to an explicitly included target."""
    positions: PositionsMetadata = {
        "schema_version": 1,
        "coordinates": {
            "spatial": {
                "axes": ["x", "y"],
                "units": "micrometer",
                "frame": frame,
                "position": "vendor cell centroid",
                "calibration": {frame: "Spatial Genomics CellCoordinates center_x/center_y in micrometres"},
            }
        },
    }
    table.uns[POSITIONS_KEY] = positions
    # The reader constructs an eager owned AnnData, never a backed Dataset2D.
    observations = cast("pd.DataFrame", table.obs)
    if region is not None:
        observations.insert(1, "region", pd.Series(region, index=observations.index, dtype="category"))
    table.obs = prepare_anndata_dataframe(observations, context=_TABLE_CONTEXT)
    table.var = prepare_anndata_dataframe(cast("pd.DataFrame", table.var), context=_TABLE_CONTEXT)
    if region is not None:
        return parse_linked_table(
            table, TableLinkage((region,), "region", "instance_id"), context=_TABLE_CONTEXT, path=counts
        )
    normalize_owned_anndata(table)
    return table
