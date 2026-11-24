"""Construct lazy, projected Xenium transcript points with known feature categories.

Parquet partitions follow row groups; their row counts come from the footer, so
partitions carry a global, monotonic row index with known divisions. A plain CSV
is split into newline-aligned blocks with a 64 MiB target.
Each block assigns unique row labels using its file-order ordinal; divisions
remain unknown. Every numeric field must hold a number, so an empty one is a
malformed record. Integer cell IDs (XOA 1.0-1.2) are signed to
keep ``-1``; they are read as text and must be decimal digits or ``-1``,
because Arrow's grammar would wrap hexadecimal ``0xffffffffffffffff`` to ``-1``.
Discovery requires a seekable source. A plain transcript CSV takes precedence
over its gzip companion when both are present.

Every task opens its source, reads only the documented columns of its partition
with their declared Arrow types, and validates it when computed: no nulls,
finite floating-point values (coordinates, ``qv``, ``nucleus_distance``),
features from the count matrix, and cell IDs from the cell population or the
unassigned sentinel (``UNASSIGNED``; ``-1`` with integer IDs). Rows keep their
source order and every ``qv``; the vendor count matrix uses only assigned
transcripts with ``qv >= 20``. Columns stay Arrow-backed, except the NumPy
coordinates ``x``, ``y``, ``z`` and the categorical ``feature_name`` that
:class:`~spatialdata.models.PointsModel` requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.csv import csv_header, read_csv_blocks
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.points import categorize_features, suppress_points_index_warning
from spatialdata_io.readers.xenium._parquet import open_parquet
from spatialdata_io.readers.xenium._schemas import (
    CELL_ID,
    STRING,
    arrow_series,
    cast_column,
    decimal_ids,
    normalize_ids,
    normalize_text,
    require_finite,
    require_no_nulls,
    transcript_columns,
    validate_membership,
    validate_parquet_types,
)

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers.xenium._layout import TabularSource
    from spatialdata_io.readers.xenium._version import Family

# Measured on 5 M rows (569 MB): streaming partitions is about 20 % faster than with
# the shared 8 MiB default, for about a third more peak memory.
_BLOCK_BYTES = 64 * 1024 * 1024
_COORDINATES = {"x_location": "x", "y_location": "y", "z_location": "z"}
_FEATURE = "feature_name"


@dataclass(frozen=True, slots=True)
class _Partitioning:
    """Immutable state shared by every partition task.

    ``cells`` holds only assigned cell IDs; the unassigned sentinel is checked separately.
    """

    source: TabularSource
    types: tuple[tuple[str, pa.DataType], ...]
    features: pd.Index
    cells: pd.Index
    unassigned: int | str
    context: ReaderErrorContext


def _empty(spec: _Partitioning) -> pa.Table:
    return pa.table({name: pa.array([], type=kind) for name, kind in spec.types})


def _normalize(table: pa.Table, spec: _Partitioning) -> pd.DataFrame:
    """Validate one projected partition and return its output columns."""
    path = spec.source.file
    require_no_nulls(table, path=path, context=spec.context)
    columns: dict[str, pd.Series | pd.Categorical | np.ndarray] = {}
    for name, kind in spec.types:
        column = table[name]
        if name == _FEATURE:
            # XOA 1.0 stores binary feature names; they are decoded strictly as UTF-8.
            names = arrow_series(normalize_text(column, name, path=path, context=spec.context))
            categories = categorize_features(names, spec.features)
            if categories is None:
                unknown = pd.Index(names.unique()).difference(spec.features, sort=False)[:5].tolist()
                raise ReaderFormatError(
                    context=spec.context,
                    path=path,
                    found=f"unknown {name} values {unknown!r}",
                    expected=f"{name} values from the count matrix",
                )
            columns[name] = categories
        elif name == CELL_ID:
            ids = arrow_series(normalize_ids(column, kind, path=path, context=spec.context))
            validate_membership(ids[ids != spec.unassigned], spec.cells, name, path=path, context=spec.context)
            columns[name] = ids
        elif pa.types.is_string(kind):
            columns[name] = arrow_series(normalize_text(column, name, path=path, context=spec.context))
        else:
            values = cast_column(column, kind, name, path=path, context=spec.context)
            if name in _COORDINATES:
                # Reuse the required NumPy coordinate vector for validation and PointsModel.
                coordinates = values.to_numpy()
                require_finite(coordinates, name, path=path, context=spec.context)
                columns[_COORDINATES[name]] = coordinates
            else:
                if pa.types.is_floating(kind):
                    require_finite(values.to_numpy(), name, path=path, context=spec.context)
                columns[name] = arrow_series(values)
    return pd.DataFrame(columns)


def _read_row_group(spec: _Partitioning, row_group: int, offset: int) -> pd.DataFrame:
    with open_parquet(spec.source.file, context=spec.context) as parquet:
        table = parquet.read_row_group(row_group, columns=[name for name, _ in spec.types])
    return _normalize(table, spec).set_axis(pd.RangeIndex(offset, offset + table.num_rows))


def _normalize_block(block: pd.DataFrame, spec: _Partitioning, *, ordinal: int, block_count: int) -> pd.DataFrame:
    """Validate a CSV block and assign row labels independent of other blocks."""
    # Arrow-backed columns convert without copying; a thread pool would only add start-up time.
    table = pa.Table.from_pandas(block, preserve_index=False, nthreads=1)
    kind = dict(spec.types)[CELL_ID]
    if pa.types.is_integer(kind):
        ids = decimal_ids(table[CELL_ID], kind, path=spec.source.file, context=spec.context)
        table = table.set_column(table.column_names.index(CELL_ID), CELL_ID, ids)
    frame = _normalize(table, spec)
    # Every block uses a distinct residue modulo block_count, even if IDs repeat.
    frame.index = pd.RangeIndex(ordinal, ordinal + len(frame) * block_count, block_count)
    return frame


def _parse(frame: dd.DataFrame, *, known_divisions: bool) -> dd.DataFrame:
    """Parse points without index or category scans."""
    with suppress_points_index_warning():
        return PointsModel.parse(
            frame,
            coordinates={"x": "x", "y": "y", "z": "z"},
            feature_key=_FEATURE,
            instance_key=CELL_ID,
            transformations={"global": Identity()},
            sort=known_divisions,
        )


def read_transcripts(
    source: TabularSource, family: Family, *, features: pd.Index, cells: pd.Index, context: ReaderErrorContext
) -> dd.DataFrame:
    """Return micrometre transcript points; construction reads only a header or footer.

    Parameters
    ----------
    source
        ``transcripts.parquet`` or a plain ``transcripts.csv``.
    family
        Identifier family of the bundle.
    features
        Count-matrix feature names in source order; the known categories of
        ``feature_name``.
    cells
        Cell IDs a transcript may be assigned to (uint64 for integer IDs).
    context
        Reader and artifact identity for header and partition failures.
    """
    path = source.file
    row_counts: list[int] = []
    if source.kind == "parquet":
        with open_parquet(path, context=context) as parquet:
            schema = parquet.schema_arrow
            names = tuple(schema.names)
            types = transcript_columns(names, source.types, path=path, context=context)
            validate_parquet_types(schema, source.types, path=path, context=context)
            row_counts = [parquet.metadata.row_group(index).num_rows for index in range(parquet.num_row_groups)]
    else:
        names = csv_header(path, context=context)
        types = transcript_columns(names, source.types, path=path, context=context)

    spec = _Partitioning(
        source, tuple(types.items()), features, cells, -1 if family == "integer_ids" else "UNASSIGNED", context
    )
    meta = _normalize(_empty(spec), spec)
    # One graph node holds the vocabularies that every partition task references.
    shared = dask.delayed(spec, pure=True)

    if source.kind == "parquet":
        groups = [(index, rows) for index, rows in enumerate(row_counts) if rows]
        if not groups:
            return _parse(dd.from_pandas(meta, npartitions=1), known_divisions=False)
        offsets = np.cumsum([0, *(rows for _, rows in groups)]).tolist()
        parts = [
            dask.delayed(_read_row_group)(shared, index, offset)
            for (index, _), offset in zip(groups, offsets, strict=False)
        ]
        divisions = (*offsets[:-1], offsets[-1] - 1)
        return _parse(dd.from_delayed(parts, meta=meta, divisions=divisions), known_divisions=True)

    # Integer cell IDs are read as text: Arrow would wrap hexadecimal in their signed type.
    blocks = read_csv_blocks(
        path,
        header=names,
        schema=pa.schema(types | {CELL_ID: STRING}),
        context=context,
        block_bytes=_BLOCK_BYTES,
    ).to_delayed()
    # Freeze the source ordinal so Dask partition selection cannot renumber rows.
    parts = [
        dask.delayed(_normalize_block)(block, shared, ordinal=ordinal, block_count=len(blocks))
        for ordinal, block in enumerate(blocks)
    ]
    return _parse(dd.from_delayed(parts, meta=meta), known_divisions=False)
