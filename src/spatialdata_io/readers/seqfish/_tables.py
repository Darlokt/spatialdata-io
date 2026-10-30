"""Bounded count parsing and exact cell alignment for seqFISH tables."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from scipy import sparse

from spatialdata_io.readers._utils.table import (
    TableLinkage,
    as_csr_array,
    normalize_owned_anndata,
    parse_linked_table,
    prepare_anndata_dataframe,
)
from spatialdata_io.readers.seqfish._schemas import (
    STRING_DTYPE,
    normalize_floats,
    normalize_ids,
    schema_error,
    validate_cell_columns,
    validate_count_columns,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.paths import ArtifactFile

_BATCH_MEMORY_BYTES = 8 * 1024 * 1024
_MAX_BATCH_ROWS = 10_000
_CSV_BLOCK_BYTES = 1024 * 1024


def _read_header(path: ArtifactFile) -> list[str]:
    """Read the CSV header before Arrow can normalize duplicate names."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
        try:
            columns = next(csv.reader([header], strict=True), list[str]())
        except csv.Error as error:
            raise schema_error(
                path, found=f"malformed CSV: {error}", expected="consistently quoted CSV records"
            ) from error
        else:
            return columns


def _count_block(batch: pa.RecordBatch, path: ArtifactFile) -> sparse.csr_array:
    """Convert a bounded lexical block to exact nonnegative integer CSR."""
    columns: list[npt.NDArray[np.int64 | np.uint64]] = []
    for column in batch.columns[1:]:
        try:
            integer_column = pc.cast(column, pa.uint64())
        except pa.ArrowInvalid:
            # Integral decimal/exponent tokens remain valid; parse them exactly
            # only when the direct integer decoder cannot handle their spelling.
            values = normalize_ids(pd.Series(column, dtype=STRING_DTYPE), path, allow_zero=True)
        else:
            values = integer_column.to_numpy()
        columns.append(values)
    largest = max(int(column.max()) for column in columns)
    dtype = np.uint64 if largest >= 2**63 else np.int64
    values = np.column_stack([column.astype(dtype, copy=False) for column in columns])
    return as_csr_array(values)


def _read_counts(path: ArtifactFile) -> tuple[sparse.csr_array, npt.NDArray[np.int64 | np.uint64], tuple[str, ...]]:
    """Keep only bounded dense buffers and accumulate canonical sparse blocks."""
    columns = _read_header(path)
    features = validate_count_columns(columns, path)
    # Bound both the source decoder's lexical buffers and the dense numeric
    # kernel. Only CSR blocks and the required cell identifiers accumulate.
    batch_rows = max(1, min(_MAX_BATCH_ROWS, _BATCH_MEMORY_BYTES // (16 * len(features))))
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
                    blocks.append(_count_block(block, path))
    except pa.ArrowInvalid as error:
        raise schema_error(
            path, found=f"malformed count CSV: {error}", expected="one field per header column"
        ) from error
    else:
        if not blocks:
            raise schema_error(path, found="no count rows", expected="a nonempty cell population")
        ids = normalize_ids(pd.Series(identifiers, dtype=STRING_DTYPE), path)
        if len(np.unique(ids)) != len(ids):
            raise schema_error(path, found="duplicate count IDs", expected="unique cell identifiers")
        # Mixed signed/unsigned batches must never promote exact counts to float.
        if any(block.dtype.kind == "u" for block in blocks):
            blocks = [block.astype(np.uint64, copy=False) for block in blocks]
        return sparse.vstack(blocks, format="csr"), ids, features


def _read_cells(path: ArtifactFile) -> pd.DataFrame:
    """Project cell metadata and validate its independent identity population."""
    columns = _read_header(path)
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
        raise schema_error(
            path, found=f"malformed cell CSV: {error}", expected="one field per header column"
        ) from error
    frame = source.to_pandas(types_mapper=pd.ArrowDtype)
    ids = normalize_ids(frame.pop("label"), path)
    if len(np.unique(ids)) != len(ids):
        raise schema_error(path, found="duplicate cell metadata IDs", expected="unique cell identifiers")
    frame.index = pd.Index(ids)
    for column in ("area", "center_x", "center_y"):
        frame[column] = normalize_floats(frame[column], path, column)
    if frame["area"].lt(0).any():
        raise schema_error(path, found="negative cell area", expected="nonnegative cell areas")
    return frame


def read_table(counts: ArtifactFile, cells: ArtifactFile) -> ad.AnnData:
    """Return an owned unlinked CSR table in count-row and feature-header order."""
    expression, identifiers, features = _read_counts(counts)
    metadata = _read_cells(cells)
    if not metadata.index.equals(pd.Index(identifiers)):
        if set(identifiers) != set(metadata.index):
            raise schema_error(
                cells, found="different count and metadata IDs", expected="identical unique cell populations"
            )
        metadata = metadata.loc[identifiers]
    observations = pd.DataFrame(index=pd.Index([str(i) for i in range(len(identifiers))]))
    observations["area"] = pd.Series(metadata["area"].array, index=observations.index)
    observations["instance_id"] = pd.Series(
        identifiers, index=observations.index, dtype=pd.ArrowDtype(pa.from_numpy_dtype(identifiers.dtype))
    )
    table = ad.AnnData(X=expression, obs=observations, var=pd.DataFrame(index=pd.Index(features)))
    table.obsm["spatial"] = metadata[["center_x", "center_y"]].to_numpy(dtype=np.float64)
    return table


def finalize_table(table: ad.AnnData, *, region: str | None, context: ReaderErrorContext) -> ad.AnnData:
    """Finalize owned annotations once and link only to an explicitly included target."""
    # This reader constructs an eager owned AnnData, never a backed Dataset2D.
    observations = cast("pd.DataFrame", table.obs)
    if region is not None:
        observations.insert(1, "region", pd.Series(region, index=observations.index, dtype="category"))
    table.obs = prepare_anndata_dataframe(observations, context=context)
    table.var = prepare_anndata_dataframe(cast("pd.DataFrame", table.var), context=context)
    if region is not None:
        return parse_linked_table(table, TableLinkage((region,), "region", "instance_id"), context=context)
    normalize_owned_anndata(table)
    return table
