"""Build projected path-backed seqFISH transcript partitions without a scan."""

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

from spatialdata_io.readers.seqfish._schemas import (
    STRING_DTYPE,
    normalize_transcript_partition,
    schema_error,
    validate_transcript_columns,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dask.delayed import Delayed

    from spatialdata_io.readers._utils.paths import ArtifactFile

_HEADER_BYTES = 1024 * 1024
_BLOCK_BYTES = 8 * 1024 * 1024


def _read_partition(
    block: bytes, *, names: tuple[str, ...], columns: tuple[str, ...], first: bool, path: ArtifactFile
) -> pd.DataFrame:
    """Decode one owned byte block with strict record width and lexical fields."""
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


def read_transcripts(path: ArtifactFile, *, features: tuple[str, ...], roi: str) -> dd.DataFrame:
    """Return lazy XY points with validated feature categories and raw z annotation.

    Only a bounded CSV header is read at construction. Known categories
    and an explicit no-sort policy prevent the public model parser from
    executing a feature or index scan. Row validation belongs to each task.
    """
    with path.open("rb") as source:
        header = source.readline(_HEADER_BYTES + 1)
    if len(header) > _HEADER_BYTES:
        raise schema_error(path, found="oversized CSV header", expected="a header at most 1 MiB")
    try:
        names = next(csv.reader([header.decode("utf-8-sig")], strict=True), [])
    except (UnicodeError, csv.Error) as error:
        raise schema_error(path, found="malformed CSV header", expected="one UTF-8 named header record") from error
    columns = validate_transcript_columns(names, path)
    # Pandas projection ignores malformed record widths. Its Arrow engine also
    # infers numbers before applying dtype, irreversibly rounding large IDs.
    # Dask owns newline block discovery and path-backed I/O; Arrow projects and
    # decodes the required fields as text inside each executed block instead.
    # Dask's read_bytes currently lacks annotations; its documented non-path
    # result is a sample plus one sequence of delayed byte blocks per file.
    read_blocks = cast("Callable[..., tuple[bytes | bool, list[list[Delayed]]]]", read_bytes)
    # The validated artifact is one literal path, even when a directory or ROI
    # name contains glob metacharacters. A list makes fsspec expand the escaped
    # pattern consistently, including bracket-only filenames.
    _, blocks = read_blocks([glob.escape(str(path))], delimiter=b"\n", blocksize=_BLOCK_BYTES, sample=False)
    parsed = [
        delayed(_read_partition)(block, names=tuple(names), columns=columns, first=i == 0, path=path)
        for i, block in enumerate(blocks[0])
    ]
    # An empty explicit frame gives the partition function the same schema as a
    # real task. It validates metadata without triggering source execution.
    empty = pd.DataFrame({name: pd.Series(dtype=STRING_DTYPE) for name in columns})
    frame = dd.from_delayed(parsed, meta=empty)
    meta = normalize_transcript_partition(empty, features=features, path=path)
    normalized = frame.map_partitions(
        normalize_transcript_partition,
        features=features,
        path=path,
        meta=meta,
    )
    # Unknown divisions are intentional for independent CSV blocks. This exact
    # upstream warning does not indicate a failed sort, since none was requested.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The index of the dataframe is not monotonic increasing.*", category=UserWarning
        )
        return PointsModel.parse(
            normalized,
            coordinates={"x": "x", "y": "y"},
            feature_key="name",
            instance_key="cell" if "cell" in columns else None,
            transformations={roi: Identity()},
            sort=False,
        )
