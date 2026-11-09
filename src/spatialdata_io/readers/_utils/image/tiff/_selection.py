"""Validate caller TIFF indexes and resolve the series a reader reads.

A TIFF can hold several image series (for example a stack and its label, or
unrelated acquisitions), and each series several pyramid levels. Readers never
pick the first of several series silently: without an explicit selection a
file must hold exactly one series. An explicit index is range-checked by
``read_tiff``, which raises ``IndexError`` with the reader's context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderFormatError

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.image.tiff._types import TiffMetadata


def validate_tiff_index(value: object, /, *, parameter: str, optional: bool) -> None:
    """Validate a caller's TIFF series or level index before any I/O.

    Parameters
    ----------
    value
        Borrowed caller value.
    parameter
        Public option name used in messages, such as ``"tiff_series"``.
    optional
        Whether ``None`` (no selection) is accepted.

    Raises
    ------
    TypeError
        If ``value`` is not an integer (booleans are rejected), or ``None``
        for a required index; the message names the type found.
    ValueError
        If ``value`` is negative.
    """
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        kind = "an integer or None" if optional else "an integer"
        message = f"{parameter} must be {kind}; found {type(value).__name__}."
        raise TypeError(message)
    if value < 0:
        message = f"{parameter} must be nonnegative; found {value}."
        raise ValueError(message)


def select_tiff_series(
    metadata: TiffMetadata,
    series: int | None,
    /,
    *,
    context: ReaderErrorContext,
    option: str | None = None,
) -> int:
    """Return the series to read: the caller's selection or the file's only series.

    Parameters
    ----------
    metadata
        Borrowed metadata of the inspected TIFF.
    series
        A validated caller selection, or ``None`` for the file's only series.
    context
        Owning reader and artifact for the ambiguity error.
    option
        Public option that selects a series, named in the error; ``None`` for
        readers without one.

    Returns
    -------
    int
        ``series`` unchanged when given, else ``0``.

    Raises
    ------
    ReaderFormatError
        If ``series`` is ``None`` and the file has several series (or none).
    """
    if series is not None:
        return series
    if len(metadata.series) != 1:
        expected = "one TIFF series" if option is None else f"one TIFF series or an explicit {option} selection"
        raise ReaderFormatError(
            context=context, path=metadata.path, found=f"{len(metadata.series)} TIFF series", expected=expected
        )
    return 0
