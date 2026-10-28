"""Construct lazy, projected Xenium transcript points with known feature categories.

Parquet partitions follow row groups; their row counts come from the footer, so
partitions carry a global, monotonic row index with known divisions. A plain CSV
is split into newline-delimited byte blocks whose indexes restart at zero. A gzip
CSV is one compressed stream that cannot be split and is rejected during
discovery.

Every task opens its source, reads only the documented columns of its partition,
and validates it when computed: finite coordinates, features from the count
matrix, and cell IDs from the cell population or the unassigned sentinel
(``UNASSIGNED``; ``-1`` with integer IDs). Rows keep their source order and every
``qv``; the vendor count matrix uses only assigned transcripts with ``qv >= 20``.
"""

from __future__ import annotations

import glob
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from dask.bytes import read_bytes
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity

from spatialdata_io.readers.xenium._schemas import (
    CELL_ID,
    cast_column,
    normalize_ids,
    normalize_text,
    require_finite,
    require_no_nulls,
    schema_error,
    text_series,
    transcript_types,
    vocabulary_codes,
)
from spatialdata_io.readers.xenium._tables import read_header

if TYPE_CHECKING:
    from collections.abc import Callable

    from dask.delayed import Delayed

    from spatialdata_io.readers.xenium._layout import TabularSource
    from spatialdata_io.readers.xenium._version import Family

_BLOCK_BYTES = 64 * 1024 * 1024
_COORDINATES = {"x_location": "x", "y_location": "y", "z_location": "z"}
_ARTIFACT = "transcripts"
_FEATURE = "feature_name"


@dataclass(frozen=True, slots=True)
class _Partitioning:
    """Immutable state shared by every partition task.

    ``cells`` holds the assigned cell IDs and the unassigned sentinel.
    """

    source: TabularSource
    family: Family
    names: tuple[str, ...]
    types: tuple[tuple[str, pa.DataType], ...]
    features: pd.Index
    cells: pd.Index


def _empty(spec: _Partitioning) -> pa.Table:
    return pa.table({name: pa.array([], type=kind) for name, kind in spec.types})


def _normalize(table: pa.Table, spec: _Partitioning) -> pd.DataFrame:
    """Validate one projected partition and return its output columns."""
    path = spec.source.file
    require_no_nulls(table, path=path, artifact=_ARTIFACT)
    columns: dict[str, pd.Series | pd.Categorical | np.ndarray] = {}
    for name, kind in spec.types:
        column = table[name]
        if name == _FEATURE:
            # XOA 1.0 stores binary feature names; they are decoded strictly as UTF-8.
            names = text_series(normalize_text(column, name, path=path, artifact=_ARTIFACT))
            codes = vocabulary_codes(names, spec.features, name, path=path, artifact=_ARTIFACT)
            columns[name] = pd.Categorical.from_codes(codes, categories=spec.features)
        elif name == CELL_ID:
            ids = normalize_ids(column, spec.family, path=path, artifact=_ARTIFACT)
            values = (
                pd.Series(np.asarray(ids.to_numpy(), dtype=np.int64))
                if spec.family == "integer_ids"
                else text_series(ids)
            )
            vocabulary_codes(values, spec.cells, name, path=path, artifact=_ARTIFACT)
            columns[name] = values
        elif pa.types.is_string(kind):
            columns[name] = text_series(normalize_text(column, name, path=path, artifact=_ARTIFACT))
        else:
            values = np.asarray(cast_column(column, kind, name, path=path, artifact=_ARTIFACT).to_numpy())
            if name in _COORDINATES:
                require_finite(values, name, path=path, artifact=_ARTIFACT)
            columns[_COORDINATES.get(name, name)] = values
    return pd.DataFrame(columns)


def _read_row_group(spec: _Partitioning, row_group: int, offset: int) -> pd.DataFrame:
    with pq.ParquetFile(spec.source.file) as parquet:
        table = parquet.read_row_group(row_group, columns=[name for name, _ in spec.types])
    return _normalize(table, spec).set_axis(pd.RangeIndex(offset, offset + table.num_rows))


def _read_block(spec: _Partitioning, block: bytes, *, first: bool) -> pd.DataFrame:
    """Parse one newline-delimited block; Arrow checks every record's full width."""
    if not block:
        # A record longer than the block size leaves the blocks it spans empty.
        return _normalize(_empty(spec), spec)
    try:
        with pa.BufferReader(block) as source:
            table = pacsv.read_csv(
                source,
                read_options=pacsv.ReadOptions(column_names=list(spec.names), skip_rows=int(first), use_threads=False),
                convert_options=pacsv.ConvertOptions(
                    include_columns=[name for name, _ in spec.types],
                    column_types=dict(spec.types),
                    strings_can_be_null=False,
                    quoted_strings_can_be_null=False,
                ),
            )
    except pa.ArrowInvalid as error:
        raise schema_error(
            spec.source.file, _ARTIFACT, found=f"malformed CSV: {error}", expected="typed records of the header width"
        ) from error
    return _normalize(table, spec)


def _parse(frame: dd.DataFrame, *, known_divisions: bool) -> dd.DataFrame:
    """Parse points without index or category scans."""
    with warnings.catch_warnings():
        # CSV blocks have unknown divisions by construction; the model warns about them.
        warnings.filterwarnings(
            "ignore", message="The index of the dataframe is not monotonic increasing.*", category=UserWarning
        )
        return PointsModel.parse(
            frame,
            coordinates={"x": "x", "y": "y", "z": "z"},
            feature_key=_FEATURE,
            instance_key=CELL_ID,
            transformations={"global": Identity()},
            sort=known_divisions,
        )


def read_transcripts(source: TabularSource, family: Family, *, features: pd.Index, cells: pd.Index) -> dd.DataFrame:
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
        Cell IDs a transcript may be assigned to (int64 for integer IDs).
    """
    path = source.file
    row_counts: list[int] = []
    if source.kind == "parquet":
        with pq.ParquetFile(path) as parquet:
            names = tuple(parquet.schema_arrow.names)
            row_counts = [parquet.metadata.row_group(index).num_rows for index in range(parquet.num_row_groups)]
    else:
        names = tuple(read_header(source))
    types = transcript_types(names, path=path)
    if family == "integer_ids":
        types |= {CELL_ID: pa.int64()}

    sentinel = pd.Index([-1]) if family == "integer_ids" else pd.Index(["UNASSIGNED"], dtype="str")
    spec = _Partitioning(source, family, names, tuple(types.items()), features, cells.append(sentinel))
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

    # read_bytes has no typed signature; this is its documented result without paths.
    blocks_of = cast("Callable[..., tuple[bytes, list[list[Delayed]]]]", read_bytes)
    _, blocks = blocks_of([glob.escape(str(path))], delimiter=b"\n", blocksize=_BLOCK_BYTES, sample=False)
    parts = [dask.delayed(_read_block)(shared, block, first=index == 0) for index, block in enumerate(blocks[0])]
    return _parse(dd.from_delayed(parts, meta=meta), known_divisions=False)
