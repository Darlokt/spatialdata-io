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
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from scipy import sparse

from spatialdata_io._positions import POSITIONS_KEY, PositionsMetadata
from spatialdata_io.readers._utils.csv import read_csv_header
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
    STRING_DTYPE,
    normalize_floats,
    normalize_ids,
    validate_cell_columns,
    validate_count_columns,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile

_TABLE_CONTEXT = ReaderErrorContext("seqfish", "table")
_BATCH_MEMORY_BYTES = 8 * 1024 * 1024
_CSV_BLOCK_BYTES = 1024 * 1024


def _count_block(batch: pa.RecordBatch, path: ArtifactFile) -> sparse.csr_array:
    """Convert a bounded lexical block to exact nonnegative integer CSR."""
    columns: list[npt.NDArray[np.int64 | np.uint64]] = []
    for column in batch.columns[1:]:
        try:
            integer_column = pc.cast(column, pa.uint64())
        except pa.ArrowInvalid:
            # Integral decimal/exponent tokens remain valid; parse them exactly
            # only when the direct integer decoder cannot handle their spelling.
            values = normalize_ids(pd.Series(column, dtype=STRING_DTYPE), path, context=COUNTS_CONTEXT, allow_zero=True)
        else:
            values = integer_column.to_numpy()
        columns.append(values)
    largest = max(int(column.max()) for column in columns)
    dtype = np.uint64 if largest >= 2**63 else np.int64
    values = np.empty((batch.num_rows, len(columns)), dtype=dtype)
    for index, column in enumerate(columns):
        values[:, index] = column
    return as_csr_array(values)


def _read_counts(path: ArtifactFile) -> tuple[sparse.csr_array, npt.NDArray[np.int64 | np.uint64], tuple[str, ...]]:
    """Keep only bounded dense buffers and accumulate canonical sparse blocks."""
    columns = read_csv_header(path, context=COUNTS_CONTEXT)
    features = validate_count_columns(columns, path)
    # Budget parsed numeric columns plus their dense destination separately
    # from Arrow text buffers, sparse conversion and retained output blocks.
    batch_rows = max(1, _BATCH_MEMORY_BYTES // (16 * len(features)))
    identifiers: list[str] = []
    blocks: list[sparse.csr_array] = []
    try:
        with (
            path.open("rb") as handle,
            pacsv.open_csv(
                handle,
                read_options=pacsv.ReadOptions(
                    column_names=columns, skip_rows=1, block_size=_CSV_BLOCK_BYTES, use_threads=False
                ),
                parse_options=pacsv.ParseOptions(newlines_in_values=True),
                convert_options=pacsv.ConvertOptions(column_types=dict.fromkeys(columns, pa.string()), null_values=[]),
            ) as reader,
        ):
            for batch in reader:
                for offset in range(0, batch.num_rows, batch_rows):
                    block = batch.slice(offset, batch_rows)
                    identifiers.extend(block.column(0).to_pylist())
                    # Rows are never summed, so a block's counts are complete
                    # once read. Finalizing it now retains float32 values
                    # instead of 8-byte integers; stacking promotes every
                    # block exactly to float64 if any block needs it, which is
                    # the dtype one final call over all blocks would select.
                    blocks.extend(
                        finalize_count_arrays((_count_block(block, path),), context=COUNTS_CONTEXT, path=path)
                    )
    except pa.ArrowInvalid as error:
        raise ReaderFormatError(
            context=COUNTS_CONTEXT,
            path=path,
            found=f"malformed CSV records ({error})",
            expected="one field per header column",
        ) from error

    if not blocks:
        raise ReaderFormatError(
            context=COUNTS_CONTEXT, path=path, found="no count rows", expected="a nonempty cell population"
        )
    ids = normalize_ids(pd.Series(identifiers, dtype=STRING_DTYPE), path, context=COUNTS_CONTEXT)
    if len(np.unique(ids)) != len(ids):
        raise ReaderFormatError(
            context=COUNTS_CONTEXT, path=path, found="duplicate count IDs", expected="unique cell identifiers"
        )
    return sparse.vstack(blocks, format="csr"), ids, features


def _read_cells(path: ArtifactFile) -> pd.DataFrame:
    """Project cell metadata and validate its independent identity population."""
    columns = read_csv_header(path, context=CELLS_CONTEXT)
    projected = validate_cell_columns(columns, path)
    try:
        with path.open("rb") as handle:
            source = pacsv.read_csv(
                handle,
                read_options=pacsv.ReadOptions(
                    column_names=columns, skip_rows=1, block_size=_CSV_BLOCK_BYTES, use_threads=False
                ),
                parse_options=pacsv.ParseOptions(newlines_in_values=True),
                convert_options=pacsv.ConvertOptions(
                    include_columns=projected, column_types=dict.fromkeys(projected, pa.string()), null_values=[]
                ),
            )
    except pa.ArrowInvalid as error:
        raise ReaderFormatError(
            context=CELLS_CONTEXT,
            path=path,
            found=f"malformed CSV records ({error})",
            expected="one field per header column",
        ) from error

    frame = source.to_pandas(types_mapper=pd.ArrowDtype)
    ids = normalize_ids(frame.pop("label"), path, context=CELLS_CONTEXT)
    if len(np.unique(ids)) != len(ids):
        raise ReaderFormatError(
            context=CELLS_CONTEXT, path=path, found="duplicate cell metadata IDs", expected="unique cell identifiers"
        )
    frame.index = pd.Index(ids)
    for column in ("area", "center_x", "center_y"):
        frame[column] = normalize_floats(frame[column], path, column, context=CELLS_CONTEXT)
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
