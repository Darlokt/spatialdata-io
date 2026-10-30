"""Construct projected, path-backed MERSCOPE transcript partitions."""

from __future__ import annotations

import csv
import glob
import warnings
from typing import TYPE_CHECKING, cast

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from dask import delayed
from dask.bytes import read_bytes
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity

from spatialdata_io.readers.merscope._schemas import (
    STRING_DTYPE,
    normalize_transcript_partition,
    schema_error,
    validate_transcript_columns,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dask.delayed import Delayed

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.merscope._layout import TranscriptSource

_HEADER_BYTES = 1024 * 1024
_BLOCK_BYTES = 8 * 1024 * 1024


def _read_partition(
    block: bytes, *, names: tuple[str, ...], columns: tuple[str, ...], first: bool, path: ArtifactFile
) -> pd.DataFrame:
    """Project lexical fields while checking the full source record width."""
    # Arrow checks full record widths but can accept malformed CSV quoting.
    try:
        with pa.BufferReader(block) as source:
            table = pacsv.read_csv(
                source,
                read_options=pacsv.ReadOptions(column_names=names, skip_rows=int(first), use_threads=False),
                parse_options=pacsv.ParseOptions(newlines_in_values=True),
                convert_options=pacsv.ConvertOptions(
                    include_columns=columns, column_types=dict.fromkeys(columns, pa.string()), null_values=[]
                ),
            )
    except pa.ArrowInvalid as error:
        raise schema_error(
            path, found=f"malformed transcript CSV: {error}", expected="one field per named header column"
        ) from error

    return table.to_pandas(types_mapper=pd.ArrowDtype)


def read_transcripts(
    source: TranscriptSource,
    *,
    features: tuple[str, ...],
    frame: str,
    allowed_cell_ids: tuple[str, ...] | None,
) -> dd.DataFrame:
    """Return calibrated XY points without reading transcript rows eagerly.

    Dask owns newline-delimited source blocks; Arrow decodes projected
    lexical fields before reader-owned value normalization. CSV records must
    each occupy one physical line. Quoted commas and quotes remain supported.
    """
    path = source.path
    with path.open("rb") as handle:
        header = handle.readline(_HEADER_BYTES + 1)
    if len(header) > _HEADER_BYTES:
        raise schema_error(path, found="oversized CSV header", expected="a header at most 1 MiB")

    try:
        names = next(csv.reader([header.decode("utf-8-sig")], strict=True), [])
    except (UnicodeError, csv.Error) as error:
        raise schema_error(path, found="malformed CSV header", expected="one UTF-8 named header record") from error
    columns = validate_transcript_columns(names, path)

    # Pandas can infer floats before applying Arrow dtypes, rounding exact IDs,
    # and projection can hide malformed record widths. Parse both inside the
    # executed block instead. read_bytes has no typed signature; this is its
    # documented result when include_path is false.
    read_blocks = cast("Callable[..., tuple[bytes | bool, list[list[Delayed]]]]", read_bytes)
    _, blocks = read_blocks([glob.escape(str(path))], delimiter=b"\n", blocksize=_BLOCK_BYTES, sample=False)
    parsed = [
        delayed(_read_partition)(block, names=tuple(names), columns=columns, first=index == 0, path=path)
        for index, block in enumerate(blocks[0])
    ]
    empty = pd.DataFrame({name: pd.Series(dtype=STRING_DTYPE) for name in columns})
    dataframe = dd.from_delayed(parsed, meta=empty)

    meta = normalize_transcript_partition(
        empty, features=features, path=path, assignment=source.assignment, allowed_cell_ids=allowed_cell_ids
    )
    normalized = dataframe.map_partitions(
        normalize_transcript_partition,
        features=features,
        path=path,
        assignment=source.assignment,
        allowed_cell_ids=allowed_cell_ids,
        meta=meta,
    )

    # Unsorted independent partitions are intentional. Known feature categories
    # and sort=False avoid the public model's eager category/index scans.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The index of the dataframe is not monotonic increasing.*", category=UserWarning
        )
        return PointsModel.parse(
            normalized,
            coordinates={"x": "x", "y": "y"},
            feature_key="gene",
            instance_key="cell_id" if source.assignment == "current" and "cell_id" in columns else None,
            transformations={frame: Identity()},
            sort=False,
        )
