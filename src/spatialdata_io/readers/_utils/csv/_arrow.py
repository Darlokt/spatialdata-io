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
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.csv as pacsv

from spatialdata_io.readers._utils.errors import ReaderFormatError, add_reader_context

if TYPE_CHECKING:
    import io
    from collections.abc import Generator
    from pathlib import Path
    from typing import IO

    from spatialdata_io.readers._utils.errors import ReaderErrorContext


def csv_header(
    path: Path, /, *, context: ReaderErrorContext, source: IO[bytes] | io.BufferedIOBase | None = None
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
        if source is None:
            with path.open("rb") as handle:
                line = handle.readline()
        else:
            line = source.readline()
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise
    with csv_errors(context=context, path=path):
        names = pacsv.read_csv(pa.py_buffer(line), read_options=pacsv.ReadOptions(use_threads=False)).column_names
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
