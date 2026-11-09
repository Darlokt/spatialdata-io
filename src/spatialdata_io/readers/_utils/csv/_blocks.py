"""Partition an uncompressed CSV lazily into newline-aligned byte blocks parsed by Arrow.

Dask splits the file into byte blocks that start and end at a newline, so
every record must occupy one physical line. Readers choose the block size
(``block_bytes``, default ``CSV_BLOCK_BYTES``): larger blocks mean fewer tasks
and faster streaming of large files, smaller blocks less memory per computed
partition. Each block task reads its byte range from the path, removes the
header line from the first block, and parses the requested columns with Arrow,
which checks that every record has the header's width even for unprojected
columns. A record longer than a block leaves the blocks it spans empty; they
become empty partitions. Building the frame reads no record; the caller has
already read the header with :func:`csv_header`.

Blocks are parsed directly with Arrow, which accepts some malformed quoting.
Columns are text unless the caller supplies Arrow types. No text is a null:
empty fields are empty strings, and an empty field in a column with a non-text
type is a malformed record. The Arrow parser retains its 1 MiB input block
size inside each Dask partition; the Dask block target does not bound the
allocation of an oversized record.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from dask import delayed
from dask.bytes import read_bytes

from spatialdata_io.readers._utils.csv._arrow import csv_errors

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from dask.delayed import Delayed

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.paths import ArtifactFile

CSV_BLOCK_BYTES = 8 * 1024 * 1024
"""Default approximate bytes per partition; one partition holds whole records only."""


@dataclass(frozen=True, slots=True)
class _BlockSpec:
    """Immutable parsing state shared by every block task of one file."""

    path: ArtifactFile
    names: tuple[str, ...]
    schema: pa.Schema
    context: ReaderErrorContext


def _empty(spec: _BlockSpec) -> pd.DataFrame:
    """Return an empty frame with the projected Arrow-backed columns of ``spec``."""
    return spec.schema.empty_table().to_pandas(types_mapper=pd.ArrowDtype)


def _read_block(block: bytes, spec: _BlockSpec, *, first: bool) -> pd.DataFrame:
    """Parse one owned byte block into the projected Arrow-backed columns."""
    if first:
        # The header is one physical line, as csv_header requires.
        newline = block.find(b"\n")
        block = block[newline + 1 :] if newline >= 0 else b""
    if not block:
        return _empty(spec)
    with pa.BufferReader(block) as source, csv_errors(context=spec.context, path=spec.path):
        table = pacsv.read_csv(
            source,
            read_options=pacsv.ReadOptions(column_names=list(spec.names), block_size=1024 * 1024, use_threads=False),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(
                include_columns=spec.schema.names,
                column_types=spec.schema,
                null_values=[],
                strings_can_be_null=False,
            ),
        )
    return table.to_pandas(types_mapper=pd.ArrowDtype)


def read_csv_blocks(  # noqa: PLR0913, RUF100 - Explicit file, schema, error origin and partition size.
    path: ArtifactFile,
    /,
    *,
    header: Sequence[str],
    columns: Sequence[str],
    context: ReaderErrorContext,
    column_types: Mapping[str, pa.DataType] | None = None,
    block_bytes: int = CSV_BLOCK_BYTES,
) -> dd.DataFrame:
    """Return projected CSV columns as lazy partitions of newline-aligned blocks.

    Parameters
    ----------
    path
        Validated uncompressed CSV file whose first physical line is ``header``.
    header
        All header names in source order, as validated by the reader.
    columns
        Header names to return, in output order.
    context
        Reader and artifact identity for malformed records, raised when a
        partition is computed.
    column_types
        Arrow types of some returned columns; others are strings.
    block_bytes
        Target block size in bytes. Like the other arguments it is trusted: a
        positive reader constant, not a public option. Larger blocks give
        fewer partitions and less scheduling overhead, but each computed
        partition holds more source bytes and parsed columns at once.

    Returns
    -------
    dask.dataframe.DataFrame
        One partition per block, in file order, with unknown divisions and a
        default index restarting in every partition. A file of ``size`` bytes
        has ``max(1, size // block_bytes)`` blocks of nearly equal size.
        Columns use :class:`pandas.ArrowDtype`.

    Notes
    -----
    Construction lists the file size only; no record is read. Every task opens
    its byte range by path and closes it.
    """
    types = column_types or {}
    schema = pa.schema([(name, types.get(name, pa.string())) for name in columns])
    spec = _BlockSpec(path, tuple(header), schema, context)
    # read_bytes has no typed signature; this is its documented result without
    # paths. Escaping keeps one literal path even with glob metacharacters.
    read_blocks = cast("Callable[..., tuple[bytes, list[list[Delayed]]]]", read_bytes)
    _, blocks = read_blocks([glob.escape(str(path))], delimiter=b"\n", blocksize=block_bytes, sample=False)
    parts = [delayed(_read_block)(block, spec, first=index == 0) for index, block in enumerate(blocks[0])]
    return dd.from_delayed(parts, meta=_empty(spec))
