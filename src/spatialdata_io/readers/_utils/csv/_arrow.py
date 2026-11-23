"""Read a CSV header with Arrow and report Arrow's CSV failures with reader context.

Readers call :mod:`pyarrow.csv` themselves with explicit Arrow types and
options; Arrow's typed CSV grammar is their contract. Two steps repeat across
readers and are not done by Arrow:

- :func:`csv_header` reads the header before the typed read, because column
  types and projections depend on it (dynamic feature columns, optional
  fields), and rejects duplicate names, which Arrow keeps and pandas renames.
- :func:`csv_errors` turns the errors Arrow raises for malformed content into
  :class:`ReaderFormatError` naming the reader, artifact and path.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, overload

import pyarrow as pa
import pyarrow.csv as pacsv

from spatialdata_io.readers._utils.csv._stream import open_csv_stream
from spatialdata_io.readers._utils.errors import ReaderFormatError, add_reader_context

if TYPE_CHECKING:
    import io
    from collections.abc import Generator
    from pathlib import Path
    from typing import IO

    from spatialdata_io.readers._utils.errors import ReaderErrorContext


_HEADER_READ_BYTES = 64 * 1024


def _header_line(source: IO[bytes] | io.BufferedIOBase, path: Path, context: ReaderErrorContext) -> bytes:
    """Read wide LF/CRLF headers incrementally and stop promptly on CR-delimited files."""
    pieces: list[bytes] = []
    while part := source.readline(_HEADER_READ_BYTES):
        carriage_return = part.find(b"\r")
        if carriage_return >= 0:
            if carriage_return == len(part) - 1:
                # CR and LF may straddle the read boundary.
                part += source.read(1)
            if carriage_return != len(part) - 2 or not part.endswith(b"\r\n"):
                raise ReaderFormatError(
                    context=context,
                    path=path,
                    found="a bare carriage return in the CSV header",
                    expected="LF or CRLF line endings",
                )
        pieces.append(part)
        if part.endswith(b"\n"):
            break
    return b"".join(pieces)


@overload
def csv_header(
    path: Path, /, *, context: ReaderErrorContext, compressed: bool = False, delimiter: str = ","
) -> tuple[str, ...]: ...


@overload
def csv_header(
    path: Path, /, *, context: ReaderErrorContext, source: IO[bytes] | io.BufferedIOBase, delimiter: str = ","
) -> tuple[str, ...]: ...


def csv_header(
    path: Path,
    /,
    *,
    context: ReaderErrorContext,
    source: IO[bytes] | io.BufferedIOBase | None = None,
    compressed: bool = False,
    delimiter: str = ",",
) -> tuple[str, ...]:
    """Parse the header record of a CSV with Arrow's grammar.

    Parameters
    ----------
    path
        The CSV artifact. It is opened and closed when ``source`` is ``None``.
    context
        Reader and artifact identity for failures.
    source
        Open binary stream of ``path``'s content positioned at the header, such
        as a gzip stream or an archive member. Exactly one physical line is
        read from it, so the caller can continue with the records.
    compressed
        Open the path as gzip when selected by the reader. A caller providing
        a stream owns its decompression.
    delimiter
        Field delimiter, such as a tab for a named TSV artifact.

    Returns
    -------
    tuple of str
        The header names in source order, as Arrow reads them: a UTF-8
        byte-order mark is removed, quoted names are unquoted, and empty names
        are kept.

    Raises
    ------
    ReaderFormatError
        If the line is not a header Arrow can parse (empty, not UTF-8, a quoted
        name spanning lines, or the last line of a file without a final
        newline), or a name occurs more than once.
    OSError
        If ``path`` cannot be read. The error gains a note naming ``context``.
    """
    try:
        owner = open_csv_stream(path, context=context, compressed=compressed) if source is None else nullcontext(source)
        with owner as stream:
            line = _header_line(stream, path, context)
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise
    with csv_errors(context=context, path=path):
        names = pacsv.read_csv(
            pa.py_buffer(line),
            read_options=pacsv.ReadOptions(use_threads=False),
            parse_options=pacsv.ParseOptions(delimiter=delimiter),
        ).column_names
    repeated = sorted(name for name, count in Counter(names).items() if count > 1)
    if repeated:
        raise ReaderFormatError(
            context=context, path=path, found=f"repeated column names {repeated!r}", expected="unique column names"
        )
    return tuple(names)


@contextmanager
def csv_errors(*, context: ReaderErrorContext, path: Path) -> Generator[None]:
    """Raise malformed CSV content found by Arrow inside the block as ``ReaderFormatError``.

    Use it around eager :func:`pyarrow.csv.read_csv` calls and around the
    iteration of :func:`pyarrow.csv.open_csv` batches.

    Parameters
    ----------
    context
        Reader and artifact identity of the CSV.
    path
        The CSV artifact.

    Yields
    ------
    None
        Control to the block that parses the CSV.

    Raises
    ------
    ReaderFormatError
        Instead of :class:`pyarrow.ArrowInvalid` (a record of the wrong width,
        a field that does not parse as its column's type, invalid UTF-8) or
        the :class:`UnicodeDecodeError` Arrow raises for a non-UTF-8 header.
        Arrow's message, which names the column and row, is kept in ``found``.
    """
    try:
        yield
    except (pa.ArrowInvalid, UnicodeDecodeError) as error:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"malformed CSV ({error})",
            expected="UTF-8 records of the header's width whose fields parse as their declared types",
        ) from error
