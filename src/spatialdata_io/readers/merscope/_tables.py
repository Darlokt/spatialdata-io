"""Build the cell table from ``cell_by_gene.csv`` and ``cell_metadata.csv``.

Counts are streamed in typed Arrow batches: each uint64 batch is converted to
a bounded dense tensor, then CSR and given its count dtype
right away, so only sparse float storage accumulates. Rows keep the count file
order and features its column order. Metadata is aligned to the count rows by
exact cell identity; both files must describe the same cells.

``Blank-<positive integer>`` controls move from ``X`` to CSR ``obsm["blank"]``
with their names in ``uns["blank_names"]``. Expression and controls share one
dtype: float32 when every count is exactly representable, else float64.
Centroids ``center_x``/``center_y`` become ``obsm["spatial"]`` (float64
micrometres, ``global`` frame). Metadata column names that a SpatialData table
does not accept are sanitized with one warning.

With boundaries returned, the table links to them through ``cells_region``
and ``EntityID``; cells whose boundaries the invalid-geometry policy dropped
are removed from the table first. Without boundaries the table is unlinked.
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
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.names import sanitize_column_names
from spatialdata_io.readers._utils.table import (
    TableLinkage,
    as_csr_array,
    finalize_count_arrays,
    normalize_owned_anndata,
    parse_linked_table,
    prepare_anndata_dataframe,
)
from spatialdata_io.readers.merscope._schemas import (
    COUNTS,
    METADATA,
    OUTPUT_ANNOTATIONS,
    STRING_DTYPE,
    blank_columns,
    count_columns,
    count_types,
    metadata_types,
    normalize_metadata,
    schema_error,
    validate_cell_ids,
    validate_output_annotations,
)

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.merscope._geometry import CellBoundaries
    from spatialdata_io.readers.merscope._layout import Names, TableFiles

_COUNTS_CONTEXT = ReaderErrorContext("merscope", COUNTS)
_METADATA_CONTEXT = ReaderErrorContext("merscope", METADATA)
_TABLE_CONTEXT = ReaderErrorContext("merscope", "cell table")
# A native batch can span the current block and the preceding partial record.
# At two source bytes per count, its uint64 value payload is at most eight
# times this block target; parser allocations, dense copies and CSR are extra.
# Records fitting one block are supported regardless of their alignment.
_BATCH_BYTES = 1024 * 1024
_PARSE = pacsv.ParseOptions(newlines_in_values=True)
# Columns whose names the reader relies on or generates; source columns that
# equal one ignoring case are renamed rather than these.
_READER_OBS_COLUMNS = ("EntityID", "center_x", "center_y", *OUTPUT_ANNOTATIONS)


def read_feature_names(path: ArtifactFile) -> tuple[str, ...]:
    """Read only the count header, including blanks in exact source order."""
    return count_columns(csv_header(path, context=_COUNTS_CONTEXT), path)


def _count_block(batch: pa.RecordBatch) -> sparse.csr_array:
    """Convert one bounded typed count batch to canonical CSR."""
    counts = batch.select(range(1, batch.num_columns)).to_tensor(row_major=True)
    return as_csr_array(np.asarray(counts))


def _read_counts(path: ArtifactFile) -> tuple[sparse.csr_array, pd.Index, tuple[str, ...]]:
    """Stream typed counts, retaining only sparse blocks and exact Arrow identities."""
    types = count_types(csv_header(path, context=_COUNTS_CONTEXT), path)
    identifiers: list[pa.Array] = []
    blocks: list[sparse.csr_array] = []
    try:
        with (
            path.open("rb") as handle,
            csv_errors(context=_COUNTS_CONTEXT, path=path),
            pacsv.open_csv(
                handle,
                read_options=pacsv.ReadOptions(block_size=_BATCH_BYTES, use_threads=False),
                parse_options=_PARSE,
                convert_options=pacsv.ConvertOptions(
                    column_types=types, include_columns=types.names, null_values=[], strings_can_be_null=False
                ),
            ) as reader,
        ):
            for batch in reader:
                identifiers.append(batch.column(0))
                # Counts are complete at ingestion. Immediate finalization retains
                # float storage; stacking promotes blocks exactly when needed.
                blocks.extend(finalize_count_arrays((_count_block(batch),), context=_COUNTS_CONTEXT, path=path))
    except OSError as error:
        add_reader_context(error, context=_COUNTS_CONTEXT, path=path)
        raise
    if not blocks:
        raise schema_error(path, COUNTS, found="no cells", expected="a nonempty count population")
    ids = pd.Series(pa.chunked_array(identifiers, pa.string()), dtype=STRING_DTYPE)
    validate_cell_ids(ids, path, COUNTS)
    if not ids.is_unique:
        raise schema_error(path, COUNTS, found="duplicate count identities", expected="one count row per exact cell ID")
    return sparse.vstack(blocks, format="csr"), pd.Index(ids, name=None), tuple(types.names[1:])


def _read_metadata(path: ArtifactFile) -> pd.DataFrame:
    """Read typed eager annotations, preserving additional source fields literally."""
    types = metadata_types(csv_header(path, context=_METADATA_CONTEXT), path)
    try:
        with path.open("rb") as handle, csv_errors(context=_METADATA_CONTEXT, path=path):
            source = pacsv.read_csv(
                handle,
                parse_options=_PARSE,
                convert_options=pacsv.ConvertOptions(
                    column_types=types, include_columns=list(types), null_values=[""], strings_can_be_null=False
                ),
            )
    except OSError as error:
        add_reader_context(error, context=_METADATA_CONTEXT, path=path)
        raise
    return normalize_metadata(source.to_pandas(types_mapper=pd.ArrowDtype), path)


def read_table(files: TableFiles) -> tuple[ad.AnnData, tuple[str, ...]]:
    """Return an owned unlinked CSR table and its complete ordered source vocabulary."""
    expression, identities, features = _read_counts(files.counts)
    observations = _read_metadata(files.metadata)
    if not observations.index.equals(identities):
        if set(observations.index) != set(identities):
            raise schema_error(
                files.metadata,
                METADATA,
                found="different cell populations",
                expected="identical count and metadata IDs",
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

    observations.columns = sanitize_column_names(
        list(observations.columns), source="merscope: the cell_metadata.csv header", reserved=_READER_OBS_COLUMNS
    )
    table = ad.AnnData(X=gene_counts, obs=observations, var=pd.DataFrame(index=pd.Index(genes)))
    table.obsm["blank"] = blank_counts
    table.uns["blank_names"] = [name for name, blank in zip(features, is_blank, strict=True) if blank]
    table.obsm["spatial"] = observations[["center_x", "center_y"]].to_numpy(dtype=np.float64)
    return table, features


def finalize_table(table: ad.AnnData, *, names: Names, boundaries: CellBoundaries | None) -> ad.AnnData:
    """Complete owned annotations and link once to an explicitly included cell footprint."""
    # The reader owns eager AnnData annotations; backed Dataset2D is not produced.
    observations = cast("pd.DataFrame", table.obs)

    expected = {"region": names.region, "slide": names.slide, "dataset_id": names.dataset_id}
    target = f"{names.dataset_id}_polygons"
    if boundaries is not None:
        expected["cells_region"] = target
    validate_output_annotations(observations, expected, context=_TABLE_CONTEXT)

    # Dropping invalid geometry must not conceal a mismatch between the sources.
    if boundaries is not None and len(boundaries.source_ids) != len(boundaries.shapes):
        if len(table.obs_names) != len(boundaries.source_ids) or not table.obs_names.isin(boundaries.source_ids).all():
            raise ReaderFormatError(
                context=_TABLE_CONTEXT,
                found="different original count and boundary populations",
                expected="identical count and original boundary IDs before geometry filtering",
            )
        table = table[table.obs_names.isin(boundaries.shapes.index)].copy()
        observations = cast("pd.DataFrame", table.obs)
        # Dropped rows may have held the only counts that needed float64.
        table.X, table.obsm["blank"] = finalize_count_arrays(
            (cast("sparse.csr_array", table.X), cast("sparse.csr_array", table.obsm["blank"])),
            context=_TABLE_CONTEXT,
        )

    positions: PositionsMetadata = {
        "schema_version": 1,
        "coordinates": {
            "spatial": {
                "axes": ["x", "y"],
                "units": "micrometer",
                "frame": "global",
                "position": "vendor cell centroid",
                "calibration": {"global": "Vizgen cell_metadata center_x/center_y in micrometres"},
            }
        },
    }
    table.uns[POSITIONS_KEY] = positions

    for name, value in expected.items():
        observations[name] = pd.Series(value, index=observations.index, dtype="category")
    observations["EntityID"] = pd.Series(observations.index, index=observations.index, dtype=STRING_DTYPE)

    table.obs = prepare_anndata_dataframe(observations, context=_TABLE_CONTEXT)
    table.var = prepare_anndata_dataframe(cast("pd.DataFrame", table.var), context=_TABLE_CONTEXT)

    if boundaries is not None:
        return parse_linked_table(
            table,
            TableLinkage((target,), "cells_region", "EntityID"),
            expected_instance_ids={target: boundaries.shapes.index},
            context=_TABLE_CONTEXT,
        )

    normalize_owned_anndata(table)
    return table
