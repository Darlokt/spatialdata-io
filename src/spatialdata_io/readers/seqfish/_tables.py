"""Read one ROI's count matrix and cell coordinates into an exact sparse AnnData table.

``<ROI>_CellxGene.csv`` holds one row per cell: the cell ID, then one count
per feature. ``<ROI>_CellCoordinates.csv`` holds the same cells, possibly in
another row order, with their area and centroid (see :mod:`._schemas`)::

    ,GeneB,GeneA                    label,area,center_x,center_y
    9,3,0                           2,12.57,4.5,1.25
    2,0,5                           9,3.14,1.5,0.5

The table keeps the count-row order and the feature-header order. Cell metadata
is aligned by exact ID, so both files must contain the same unique IDs. Counts
are read in bounded Arrow batches and stored as canonical CSR; the dense
expression matrix is never allocated. ``area`` (µm²) becomes an observation
column, ``(center_x, center_y)`` (µm) becomes ``obsm["spatial"]``, and the
cell ID becomes ``obs["instance_id"]``. Observation names are positional
strings starting at ``"0"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import sparse

from spatialdata_io._positions import POSITIONS_KEY, PositionsMetadata
from spatialdata_io.readers._utils.csv import (
    decode_float64,
    decode_integer_block,
    decode_integers,
    iter_csv_batches,
    read_csv_header,
    read_csv_table,
)
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
    CELLS_CONTEXT,
    COUNTS_CONTEXT,
    FLOAT_DTYPE,
    validate_cell_columns,
    validate_count_columns,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile

_TABLE_CONTEXT = ReaderErrorContext("seqfish", "table")
_BATCH_MEMORY_BYTES = 8 * 1024 * 1024


def _count_block(batch: pa.RecordBatch, features: tuple[str, ...], path: ArtifactFile) -> sparse.csr_array:
    """Convert a bounded lexical block to exact nonnegative integer CSR."""
    counts = decode_integer_block(batch.columns[1:], names=features, minimum=0, context=COUNTS_CONTEXT, path=path)
    return as_csr_array(counts)


def _read_counts(path: ArtifactFile) -> tuple[sparse.csr_array, npt.NDArray[np.int64 | np.uint64], tuple[str, ...]]:
    """Keep only bounded dense buffers and accumulate canonical sparse blocks."""
    columns = read_csv_header(path, context=COUNTS_CONTEXT)
    features = validate_count_columns(columns, path)
    # 16 bytes per count: its 8-byte decoded value and its concatenated text
    # with offsets, apart from Arrow's parser batches and the sparse output.
    batch_rows = max(1, _BATCH_MEMORY_BYTES // (16 * len(features)))
    identifiers: list[pa.Array] = []
    blocks: list[sparse.csr_array] = []
    with path.open("rb") as handle:
        for batch in iter_csv_batches(handle, header=columns, context=COUNTS_CONTEXT, path=path):
            for offset in range(0, batch.num_rows, batch_rows):
                block = batch.slice(offset, batch_rows)
                identifiers.append(block.column(0))
                # Rows are never summed, so a block's counts are complete
                # once read. Finalizing it now retains float32 values
                # instead of 8-byte integers; stacking promotes every
                # block exactly to float64 if any block needs it, which is
                # the dtype one final call over all blocks would select.
                blocks.extend(
                    finalize_count_arrays((_count_block(block, features, path),), context=COUNTS_CONTEXT, path=path)
                )

    if not blocks:
        raise ReaderFormatError(
            context=COUNTS_CONTEXT, path=path, found="no count rows", expected="a nonempty cell population"
        )
    ids = decode_integers(
        pa.chunked_array(identifiers, pa.string()), name="cell ID", minimum=1, context=COUNTS_CONTEXT, path=path
    )
    if len(np.unique(ids)) != len(ids):
        raise ReaderFormatError(
            context=COUNTS_CONTEXT, path=path, found="duplicate count IDs", expected="unique cell identifiers"
        )
    return sparse.vstack(blocks, format="csr"), ids, features


def _read_cells(path: ArtifactFile) -> pd.DataFrame:
    """Project cell metadata and validate its independent identity population."""
    columns = read_csv_header(path, context=CELLS_CONTEXT)
    projected = validate_cell_columns(columns, path)
    with path.open("rb") as handle:
        source = read_csv_table(handle, header=columns, context=CELLS_CONTEXT, path=path, columns=projected)

    ids = decode_integers(source.column("label"), name="label", minimum=1, context=CELLS_CONTEXT, path=path)
    if len(np.unique(ids)) != len(ids):
        raise ReaderFormatError(
            context=CELLS_CONTEXT, path=path, found="duplicate cell metadata IDs", expected="unique cell identifiers"
        )
    frame = pd.DataFrame(
        {
            column: pd.Series(
                decode_float64(source.column(column), name=column, context=CELLS_CONTEXT, path=path), dtype=FLOAT_DTYPE
            )
            for column in ("area", "center_x", "center_y")
        }
    ).set_axis(pd.Index(ids))
    if frame["area"].lt(0).any():
        raise ReaderFormatError(
            context=CELLS_CONTEXT, path=path, found="negative cell area", expected="nonnegative cell areas"
        )
    return frame


def read_table(counts: ArtifactFile, cells: ArtifactFile) -> ad.AnnData:
    """Return an owned unlinked CSR table in count-row and feature-header order."""
    expression, identifiers, features = _read_counts(counts)
    metadata = _read_cells(cells)
    count_index = pd.Index(identifiers)
    if not metadata.index.equals(count_index):
        positions = metadata.index.get_indexer(count_index)
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
    observations["instance_id"] = pd.Series(
        identifiers, index=observations.index, dtype=pd.ArrowDtype(pa.from_numpy_dtype(identifiers.dtype))
    )
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
