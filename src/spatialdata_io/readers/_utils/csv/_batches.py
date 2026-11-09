"""Parse an open CSV stream eagerly into Arrow text record batches or one table.

The caller owns a binary stream positioned at the header record, such as a
plain file or a gzip stream, and has parsed that record strictly with
:func:`parse_csv_header` or :func:`read_csv_header`. Arrow skips it and parses
every following record as text with one configuration, which the lazy block
reader :func:`read_csv_blocks` shares:

- every field is a string and no text is a null, so an empty field is ``""``;
- quoted fields may contain delimiters, doubled quotes and newlines;
- every record must have exactly as many fields as the header, including
  fields that are not returned. Arrow does not reject every malformed quote:
  a quote inside an unquoted field is literal text.

Arrow reads about ``block_bytes`` of source text per record batch (1 MiB by
default), and the header record and every record must each fit in one block:
a longer record fails as malformed. Readers decode numbers
(:func:`decode_float64`, :func:`decode_integers`) and choose how many rows
they decode at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.csv as pacsv

from spatialdata_io.readers._utils.errors import ReaderFormatError

if TYPE_CHECKING:
    import io
    from collections.abc import Generator, Sequence
    from pathlib import Path

    from spatialdata_io.readers._utils.errors import ReaderErrorContext

CSV_BATCH_BYTES = 1024 * 1024
"""Default bytes of source text per Arrow record batch."""


def iter_csv_batches(  # noqa: PLR0913, RUF100 - Explicit stream, header, error origin and batch size.
    source: io.BufferedIOBase,
    /,
    *,
    header: Sequence[str],
    context: ReaderErrorContext,
    path: Path,
    block_bytes: int = CSV_BATCH_BYTES,
) -> Generator[pa.RecordBatch]:
    """Stream the records after the header as text record batches.

    Parameters
    ----------
    source
        Open binary stream positioned at the header record. The caller owns
        it and closes it; the iterator only reads from it.
    header
        All header names in source order, as validated by the reader.
    context
        Reader and artifact identity for malformed records.
    path
        Artifact that ``source`` reads, for malformed records.
    block_bytes
        Approximate bytes of source text per batch. Like the other arguments
        it is trusted: a positive reader constant, not a public option.

    Yields
    ------
    pyarrow.RecordBatch
        Consecutive records in file order, one string column per header name.
        Closing the generator early closes Arrow's reader.

    Raises
    ------
    ReaderFormatError
        If a record does not have one field per header column or does not fit
        in one block, raised when its batch is reached.
    """
    read, parse, convert = _options(header, _text_schema(header), skip_rows=1, block_bytes=block_bytes)
    try:
        with pacsv.open_csv(source, read_options=read, parse_options=parse, convert_options=convert) as reader:
            yield from reader
    except pa.ArrowInvalid as error:
        raise _malformed(error, context=context, path=path) from error


def read_csv_table(
    source: io.BufferedIOBase,
    /,
    *,
    header: Sequence[str],
    context: ReaderErrorContext,
    path: Path,
    columns: Sequence[str] | None = None,
) -> pa.Table:
    """Read the records after the header as one text table.

    Parameters
    ----------
    source
        Open binary stream positioned at the header record. The caller owns
        it and closes it.
    header
        All header names in source order, as validated by the reader.
    context
        Reader and artifact identity for malformed records.
    path
        Artifact that ``source`` reads, for malformed records.
    columns
        Header names to return, in output order; every column by default.
        Unreturned fields still count towards each record's width.

    Returns
    -------
    pyarrow.Table
        One string column per returned name, the rows in file order.

    Raises
    ------
    ReaderFormatError
        If a record does not have one field per header column or does not fit
        in one block of ``CSV_BATCH_BYTES``.
    """
    schema = _text_schema(header if columns is None else columns)
    return read_records(source, header, schema, skip_rows=1, context=context, path=path)


def read_records(  # noqa: PLR0913, RUF100 - Explicit source, header, projection, header skip and error origin.
    source: io.BufferedIOBase | pa.NativeFile,
    header: Sequence[str],
    schema: pa.Schema,
    *,
    skip_rows: int,
    context: ReaderErrorContext,
    path: Path,
) -> pa.Table:
    """Parse ``schema``'s columns of a whole CSV source with the shared options.

    ``skip_rows`` is 1 when ``source`` still starts with the header record.
    """
    read, parse, convert = _options(header, schema, skip_rows=skip_rows, block_bytes=CSV_BATCH_BYTES)
    try:
        return pacsv.read_csv(source, read_options=read, parse_options=parse, convert_options=convert)
    except pa.ArrowInvalid as error:
        raise _malformed(error, context=context, path=path) from error


def _text_schema(names: Sequence[str]) -> pa.Schema:
    """Return string fields for ``names``."""
    return pa.schema([(name, pa.string()) for name in names])


def _options(
    header: Sequence[str], schema: pa.Schema, *, skip_rows: int, block_bytes: int
) -> tuple[pacsv.ReadOptions, pacsv.ParseOptions, pacsv.ConvertOptions]:
    """Return the one Arrow configuration: named fields, projected types, no null text."""
    return (
        pacsv.ReadOptions(column_names=list(header), skip_rows=skip_rows, block_size=block_bytes, use_threads=False),
        pacsv.ParseOptions(newlines_in_values=True),
        pacsv.ConvertOptions(include_columns=schema.names, column_types=schema, null_values=[]),
    )


def _malformed(error: pa.ArrowInvalid, *, context: ReaderErrorContext, path: Path) -> ReaderFormatError:
    """Describe a record that Arrow could not split into the header's fields."""
    return ReaderFormatError(
        context=context,
        path=path,
        found=f"malformed CSV records ({error})",
        expected="records with one field per header column",
    )
