"""Partition an uncompressed CSV lazily into newline-aligned byte blocks parsed by Arrow.

Dask splits the file into byte blocks that start and end at a newline, so
every record must occupy one physical line. Readers choose the block size
(``block_bytes``, default ``CSV_BLOCK_BYTES``): larger blocks mean fewer tasks
and faster streaming of large files, smaller blocks less memory per computed
partition. Each block task reads its byte range from the path, removes the
header line from the first block, checks the quotes of every line, and parses
the requested columns with Arrow, which checks that every record has the
header's width even for unprojected columns. A record longer than a block
leaves the blocks it spans empty; they become empty partitions. Building the
frame reads no record; the caller has already read the header with
:func:`csv_header`.

A line with an odd number of double quotes is malformed. This rejects quoted
fields spanning lines, whose records Dask would split at whichever block
boundary falls inside them, and also a single stray quote inside an unquoted
field (``5" tall``). Within a line Arrow's quoting applies: quoted delimiters
and doubled quotes are read, and some malformed quoting is accepted, such as
two stray quotes kept literally. A line that combines a stray quote with a
quoted field left open is malformed CSV with an undefined result.

Columns are text unless the caller supplies Arrow types. No text is a null:
empty fields are empty strings, and an empty field in a column with a non-text
type is a malformed record. Arrow uses a 1 MiB parser block, enlarged for an
individual long record so its position in the partition does not change whether
it can be decoded. The Dask block target does not bound the allocation of an
oversized record.
"""

from __future__ import annotations

import codecs
import glob
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from dask import delayed
from dask.bytes import read_bytes

from spatialdata_io.readers._utils.csv._arrow import csv_errors
from spatialdata_io.readers._utils.errors import ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from dask.delayed import Delayed

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.paths import ArtifactFile

CSV_BLOCK_BYTES = 8 * 1024 * 1024
"""Default approximate bytes per partition; one partition holds whole records only."""

_ARROW_BLOCK_BYTES = 1024 * 1024
_QUOTE, _NEWLINE = ord('"'), ord("\n")
# Quotes are counted in cache-sized slices: NumPy releases the GIL, so threaded
# partitions check their blocks in parallel.
_QUOTE_SLICE_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class _BlockSpec:
    """Immutable parsing state shared by every block task of one file."""

    path: ArtifactFile
    names: tuple[str, ...]
    schema: pa.Schema
    context: ReaderErrorContext


def _empty(schema: pa.Schema) -> pd.DataFrame:
    """Return an empty frame with the projected Arrow-backed columns of ``schema``."""
    return schema.empty_table().to_pandas(types_mapper=pd.ArrowDtype)


def _paired_quotes(block: bytes | bytearray) -> bool:
    """Return whether every line of a block has an even number of double quotes."""
    if b'"' not in block:
        return True
    data = np.frombuffer(block, dtype=np.uint8)
    quotes = 0
    for offset in range(0, len(data), _QUOTE_SLICE_BYTES):
        part = data[offset : offset + _QUOTE_SLICE_BYTES]
        marks = part[(part == _QUOTE) | (part == _NEWLINE)]
        newlines = np.flatnonzero(marks == _NEWLINE)
        # Every line is even exactly when the quotes before each newline (those
        # of earlier slices plus its position among the marks minus the
        # newlines before it) and the quotes in total are even.
        if ((quotes + newlines - np.arange(len(newlines))) % 2).any():
            return False
        quotes += len(marks) - len(newlines)
    return quotes % 2 == 0


def _csv_data_buffer(
    block: bytes | bytearray, /, *, context: ReaderErrorContext, path: ArtifactFile
) -> bytes | bytearray:
    """Prepare headerless physical CSV records for Arrow without changing field values.

    A partition owner calls this on a complete record block before Arrow
    parsing so the validity of a record cannot depend on block boundaries.
    The input buffer is reused unless a literal initial U+FEFF needs protection.
    """
    if b"\r" in block and re.search(rb"\r(?!\n)", block):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="a bare carriage return in CSV records",
            expected="LF or CRLF line endings",
        )
    if not _paired_quotes(block):
        start = 0
        while start < len(block):
            end = block.find(b"\n", start)
            end = len(block) if end < 0 else end
            if block.count(b'"', start, end) % 2:
                preview = block[start : min(end, start + 80)].decode(errors="replace")
                raise ReaderFormatError(
                    context=context,
                    path=path,
                    found=f"a line with an odd number of double quotes ({preview!r})",
                    expected="one record per line with paired quotes; quoted fields cannot span lines",
                )
            start = end + 1
    # Arrow treats the start of each input as a possible file BOM. A blank
    # record protects a literal U+FEFF in the first field of a data block.
    return b"\n" + block if block.startswith(codecs.BOM_UTF8) else block


def _parser_block_size(records: bytes | bytearray) -> int:
    """Size Arrow blocks to fit whole records regardless of partition alignment."""
    size = boundary = _ARROW_BLOCK_BYTES
    while boundary < len(records):
        # Only a record crossing a nominal boundary can exceed the default.
        # Jump past it so even a record spanning many blocks is visited once.
        start = records.rfind(b"\n", 0, boundary) + 1
        newline = records.find(b"\n", boundary)
        end = len(records) if newline == -1 else newline + 1
        size = max(size, end - start)
        boundary = (end // _ARROW_BLOCK_BYTES + 1) * _ARROW_BLOCK_BYTES
    return size


def parse_csv_records(
    block: bytes | bytearray,
    /,
    *,
    header: Sequence[str],
    schema: pa.Schema,
    context: ReaderErrorContext,
    path: ArtifactFile,
) -> pd.DataFrame:
    """Parse a headerless block of complete physical CSV records for a lazy points partition.

    The owner supplies the source header and projected Arrow schema. Empty text
    stays literal, and malformed records fail with their source context. The
    returned columns are Arrow-backed, including for an empty partition.
    """
    if not block:
        return _empty(schema)
    records = _csv_data_buffer(block, context=context, path=path)
    size = _parser_block_size(records)
    try:
        options = pacsv.ReadOptions(column_names=list(header), block_size=size, use_threads=False)
    except OverflowError as error:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"a {size}-byte CSV record",
            expected="a record fitting an Arrow parser block",
        ) from error
    with pa.BufferReader(records) as source, csv_errors(context=context, path=path):
        table = pacsv.read_csv(
            source,
            read_options=options,
            convert_options=pacsv.ConvertOptions(
                include_columns=schema.names,
                column_types=schema,
                null_values=[],
                strings_can_be_null=False,
            ),
        )
    return table.to_pandas(types_mapper=pd.ArrowDtype)


def _read_block(block: bytes, spec: _BlockSpec, *, first: bool) -> pd.DataFrame:
    """Drop the file header in the first partition and decode the projected records."""
    if first:
        newline = block.find(b"\n")
        block = block[newline + 1 :] if newline >= 0 else b""
    return parse_csv_records(block, header=spec.names, schema=spec.schema, context=spec.context, path=spec.path)


def read_csv_blocks(  # noqa: PLR0913, RUF100 - Explicit file, schema, error origin and partition size.
    path: ArtifactFile,
    /,
    *,
    header: Sequence[str],
    schema: pa.Schema,
    context: ReaderErrorContext,
    block_bytes: int = CSV_BLOCK_BYTES,
) -> dd.DataFrame:
    """Return projected CSV columns as lazy partitions of newline-aligned blocks.

    Parameters
    ----------
    path
        Validated uncompressed CSV file whose first physical line is ``header``.
    header
        All header names in source order, as validated by the reader.
    schema
        Projected header names and their Arrow types in output order.
    context
        Reader and artifact identity for malformed records, raised when a
        partition is computed.
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
    its byte range by path and closes it. Computing a partition raises
    :class:`ReaderFormatError` for unpaired quotes, bare carriage returns or
    malformed records. LF and CRLF delimit physical lines. Only a BOM in the
    file header is an encoding marker.
    """
    spec = _BlockSpec(path, tuple(header), schema, context)
    # read_bytes has no typed signature; this is its documented result without
    # paths. Escaping keeps one literal path even with glob metacharacters.
    read_blocks = cast("Callable[..., tuple[bytes, list[list[Delayed]]]]", read_bytes)
    _, blocks = read_blocks([glob.escape(str(path))], delimiter=b"\n", blocksize=block_bytes, sample=False)
    parts = [delayed(_read_block)(block, spec, first=index == 0) for index, block in enumerate(blocks[0])]
    return dd.from_delayed(parts, meta=_empty(schema))
