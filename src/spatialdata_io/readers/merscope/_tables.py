"""Bounded MERSCOPE counts, aligned owned annotations, and final table linkage."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.table import (
    TableLinkage,
    as_csr_array,
    normalize_owned_anndata,
    parse_linked_table,
    prepare_anndata_dataframe,
)
from spatialdata_io.readers.merscope._schemas import (
    STRING_DTYPE,
    blank_columns,
    count_columns,
    integer_values,
    metadata_columns,
    normalize_cell_ids,
    normalize_metadata,
    schema_error,
    validate_output_annotations,
)

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile

_MAX_HEADER_CHARACTERS = 1_048_576
_CSV_BLOCK_BYTES = 1024 * 1024
_BATCH_MEMORY_BYTES = 8 * 1024 * 1024
_MAX_BATCH_ROWS = 10_000


def _read_header(path: ArtifactFile) -> list[str]:
    """Bound the first physical UTF-8 CSV record before interpreting names."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = handle.readline(_MAX_HEADER_CHARACTERS + 1)
    if len(header) > _MAX_HEADER_CHARACTERS:
        raise schema_error(path, found="oversized CSV header", expected="a header of at most 1 Mi characters")
    try:
        columns = next(csv.reader([header], strict=True), list[str]())
    except csv.Error as error:
        raise schema_error(
            path, found="malformed CSV header", expected="one consistently quoted header record"
        ) from error
    else:
        return columns


def read_feature_names(path: ArtifactFile) -> tuple[str, ...]:
    """Read only the bounded count header, including blanks in exact source order."""
    return count_columns(_read_header(path), path)


def _count_block(batch: pa.RecordBatch, path: ArtifactFile) -> sparse.csr_array:
    """Decode a bounded lexical count block without float promotion or dense accumulation."""
    columns = [integer_values(pd.Series(column, dtype=STRING_DTYPE), path) for column in batch.columns[1:]]
    dtype = np.uint64 if any(column.dtype.kind == "u" for column in columns) else np.int64
    dense = np.column_stack([column.astype(dtype, copy=False) for column in columns])
    return as_csr_array(dense)


def _read_counts(path: ArtifactFile) -> tuple[sparse.csr_array, pd.Index, tuple[str, ...]]:
    """Stream exact counts, retaining only sparse blocks and required row identities."""
    columns = _read_header(path)
    features = count_columns(columns, path)
    rows_per_batch = max(1, min(_MAX_BATCH_ROWS, _BATCH_MEMORY_BYTES // (16 * len(features))))
    blocks: list[sparse.csr_array] = []
    identifiers: list[str] = []
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
                for offset in range(0, batch.num_rows, rows_per_batch):
                    block = batch.slice(offset, rows_per_batch)
                    identifiers.extend(block.column(0).to_pylist())
                    blocks.append(_count_block(block, path))
    except pa.ArrowInvalid as error:
        raise schema_error(
            path, found=f"malformed count CSV: {error}", expected="one field per header column"
        ) from error
    else:
        if not blocks:
            raise schema_error(path, found="no cells", expected="a nonempty count population")
        ids = normalize_cell_ids(pd.Series(identifiers, dtype=STRING_DTYPE), path)
        if not ids.is_unique:
            raise schema_error(path, found="duplicate count identities", expected="one count row per exact cell ID")
        if any(block.dtype.kind == "u" for block in blocks):
            blocks = [block.astype(np.uint64, copy=False) for block in blocks]
        return sparse.vstack(blocks, format="csr"), pd.Index(ids.to_numpy(), name=None), features


def _read_metadata(path: ArtifactFile) -> pd.DataFrame:
    """Read finite eager annotations once, preserving extra source fields as text."""
    columns = metadata_columns(_read_header(path), path)
    try:
        with path.open("rb") as handle:
            source = pacsv.read_csv(
                handle,
                read_options=pacsv.ReadOptions(
                    column_names=columns, skip_rows=1, block_size=_CSV_BLOCK_BYTES, use_threads=False
                ),
                parse_options=pacsv.ParseOptions(newlines_in_values=True),
                convert_options=pacsv.ConvertOptions(column_types=dict.fromkeys(columns, pa.string()), null_values=[]),
            )
    except pa.ArrowInvalid as error:
        raise schema_error(
            path, found=f"malformed metadata CSV: {error}", expected="one field per header column"
        ) from error
    else:
        return normalize_metadata(source.to_pandas(types_mapper=pd.ArrowDtype), path)


def read_table(counts: ArtifactFile, metadata: ArtifactFile) -> tuple[ad.AnnData, tuple[str, ...]]:
    """Return an owned unlinked CSR table and its complete ordered source vocabulary."""
    expression, identities, features = _read_counts(counts)
    observations = _read_metadata(metadata)
    if not observations.index.equals(identities):
        if set(observations.index) != set(identities):
            raise schema_error(
                metadata, found="different cell populations", expected="identical count and metadata IDs"
            )
        observations = observations.loc[identities]

    is_blank = np.asarray(blank_columns(features), dtype=np.bool_)
    genes = [name for name, blank in zip(features, is_blank, strict=True) if not blank]
    if not is_blank.any():
        gene_counts = expression
        blank_counts = sparse.csr_array((len(identities), 0), dtype=expression.dtype)
    elif is_blank.all():
        gene_counts = sparse.csr_array((len(identities), 0), dtype=expression.dtype)
        blank_counts = expression
    else:
        gene_counts = expression[:, ~is_blank]
        blank_counts = expression[:, is_blank]

    table = ad.AnnData(X=gene_counts, obs=observations, var=pd.DataFrame(index=pd.Index(genes)))
    table.obsm["blank"] = blank_counts
    table.uns["blank_names"] = [name for name, blank in zip(features, is_blank, strict=True) if blank]
    table.obsm["spatial"] = observations[["center_x", "center_y"]].to_numpy(dtype=np.float64)
    return table, features


def finalize_table(
    table: ad.AnnData,
    *,
    region_name: str,
    slide_name: str,
    dataset_id: str,
    target: str | None,
    expected_instances: pd.Index | None,
) -> ad.AnnData:
    """Complete owned annotations and link once to an explicitly included cell footprint."""
    context = ReaderErrorContext("merscope", "cell table")
    # The reader owns eager AnnData annotations; backed Dataset2D is not produced.
    observations = cast("pd.DataFrame", table.obs)
    expected = {"region": region_name, "slide": slide_name, "dataset_id": dataset_id}
    if target is not None:
        expected["cells_region"] = target
    validate_output_annotations(observations, expected, context=context)
    for name, value in (("region", region_name), ("slide", slide_name), ("dataset_id", dataset_id)):
        observations[name] = pd.Series(value, index=observations.index, dtype="category")
    if target is not None:
        observations["cells_region"] = pd.Series(target, index=observations.index, dtype="category")
    observations["EntityID"] = pd.Series(observations.index, index=observations.index, dtype=STRING_DTYPE)
    table.obs = prepare_anndata_dataframe(observations, context=context)
    table.var = prepare_anndata_dataframe(cast("pd.DataFrame", table.var), context=context)
    if target is not None:
        return parse_linked_table(
            table,
            TableLinkage((target,), "cells_region", "EntityID"),
            expected_instance_ids={target: expected_instances} if expected_instances is not None else None,
            context=context,
        )
    normalize_owned_anndata(table)
    return table
