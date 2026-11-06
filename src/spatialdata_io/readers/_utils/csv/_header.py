"""Parse the first physical record of a UTF-8 CSV strictly, before any column is interpreted.

Readers check header names (duplicates, required fields, dynamic feature
columns) before a parser such as Arrow or pandas can rename or drop them. The
header is one physical line: a UTF-8 byte-order mark is removed, quoting is
strict, and a quoted newline inside a header name is malformed.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderFormatError, add_reader_context

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.paths import ArtifactFile

_EXPECTED = "one strictly quoted UTF-8 CSV header record"


def parse_csv_header(
    line: bytes,
    /,
    *,
    context: ReaderErrorContext,
    path: Path,
    delimiter: str = ",",
) -> list[str]:
    """Parse one raw header line read by the caller from its own source.

    Parameters
    ----------
    line
        First physical line, as returned by ``readline()`` on a binary stream
        such as a gzip member or archive entry.
    context
        Reader and artifact identity for malformed headers.
    path
        Artifact that stores the line, for malformed headers.
    delimiter
        Field delimiter.

    Returns
    -------
    list of str
        Header names in source order; empty for an empty line.

    Raises
    ------
    ReaderFormatError
        If the line is not UTF-8 or not one strictly quoted CSV record.
    """
    try:
        return next(csv.reader([line.decode("utf-8-sig")], delimiter=delimiter, strict=True), [])
    except UnicodeDecodeError as error:
        raise ReaderFormatError(
            context=context, path=path, found="malformed CSV header (not valid UTF-8)", expected=_EXPECTED
        ) from error
    except csv.Error as error:
        raise ReaderFormatError(
            context=context, path=path, found=f"malformed CSV header ({error})", expected=_EXPECTED
        ) from error


def read_csv_header(path: ArtifactFile, /, *, context: ReaderErrorContext, delimiter: str = ",") -> list[str]:
    """Read and strictly parse the header of an uncompressed CSV file.

    Parameters
    ----------
    path
        Validated regular file. Only its first physical line is read.
    context
        Reader and artifact identity for failures.
    delimiter
        Field delimiter.

    Returns
    -------
    list of str
        Header names in source order; empty for an empty file.

    Raises
    ------
    ReaderFormatError
        If the header is not UTF-8 or not one strictly quoted CSV record.
    OSError
        If the file cannot be read. The error gains a note naming ``context``.
    """
    try:
        with path.open("rb") as source:
            line = source.readline()
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise
    return parse_csv_header(line, context=context, path=path, delimiter=delimiter)
