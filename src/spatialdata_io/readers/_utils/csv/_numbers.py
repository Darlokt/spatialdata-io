"""Decode exact numbers from the text of CSV fields, failing instead of losing precision.

Readers parse CSV fields as text and decode numeric columns from their source
spelling, so that no value silently changes on the way:

- :func:`decode_float64` returns correctly rounded, finite float64 values. A
  nonzero spelling that rounds to zero (``1e-400``) fails, and so does an
  integral value of at least 2**53 that float64 cannot hold exactly
  (``9007199254740993``, or ``1e300``, which is not exactly ``10**300`` in
  float64); rounding it would silently change an integer.
- :func:`decode_integers` returns exact int64 values, or uint64 when a
  nonnegative column needs it. Besides plain decimal integers it accepts every
  exact decimal spelling of an integer (``+2``, ``2.0``, ``3e0``, ``007``).
  Fractions, hexadecimal (which Arrow's integer cast would accept), whitespace,
  digit separators and values outside the 64-bit range fail.

:func:`decode_integer_block` and :func:`decode_float64_block` decode the
equally long columns of a bounded rows-by-columns block, such as a count
matrix batch, in one pass over their concatenated text; only an invalid block
is decoded again column by column, so that the error names its column.

Arrow's vectorized parsers decode the common spellings; exact
:class:`~decimal.Decimal` arithmetic checks only the rare values whose decoding
could hide a change. Column selection, missing-value spellings, the dtype a
reader stores and the column names in messages stay with each reader.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from spatialdata_io.readers._utils.errors import ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import numpy.typing as npt

    from spatialdata_io.readers._utils.errors import ReaderErrorContext

LITERAL_ZERO = r"[+-]?(?:0+(?:\.0*)?|\.0+)(?:[eE][+-]?\d+)?"
"""Regular expression matching every decimal spelling of zero, such as ``-0.00`` or ``.0e5``."""

# Every decimal spelling of a number, ASCII digits only.
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
# Smallest magnitude at which float64 no longer holds every integer exactly.
_FLOAT64_EXACT_LIMIT = 2**53


def decode_float64(
    text: pa.Array | pa.ChunkedArray, /, *, name: str, context: ReaderErrorContext, path: Path
) -> npt.NDArray[np.float64]:
    """Decode text into correctly rounded float64 values without silent loss.

    Parameters
    ----------
    text
        Arrow string values, one per row.
    name
        Column name used in error messages.
    context
        Reader and artifact identity for invalid values.
    path
        Artifact that stores the values, for invalid values.

    Returns
    -------
    numpy.ndarray
        One float64 value per row, the correctly rounded value of its spelling.
        It is a read-only view of Arrow memory when ``text`` is one array.

    Raises
    ------
    ReaderFormatError
        If a value is missing, not a decimal number, not finite (``inf``,
        ``nan`` or out of the float64 range), a nonzero spelling that rounds
        to zero, or an integer of at least 2**53 that float64 cannot hold
        exactly.
    """
    expected = f"finite float64 {name!r} values without underflow or integer precision loss"
    if text.null_count:
        raise ReaderFormatError(context=context, path=path, found=f"missing {name!r} values", expected=expected)
    try:
        values = pc.cast(text, pa.float64()).to_numpy(zero_copy_only=False)
    except pa.ArrowInvalid as error:
        raise ReaderFormatError(
            context=context, path=path, found=f"nonnumeric {name!r} values", expected=expected
        ) from error
    # NaN propagates through min and max, so the extremes decide finiteness.
    lowest, highest = float(values.min(initial=0.0)), float(values.max(initial=0.0))
    if not (math.isfinite(lowest) and math.isfinite(highest)):
        raise ReaderFormatError(context=context, path=path, found=f"nonfinite {name!r} values", expected=expected)

    # Arrow rounds every decimal spelling correctly. Only a nonzero spelling
    # that became zero, or a large integer beyond float64's significand, can
    # hide a change; only those rare values are compared exactly.
    zero = values == 0
    suspect = zero.copy()
    if zero.any():
        suspect[zero] = ~_full_match(text.filter(pa.array(zero)), LITERAL_ZERO)
    if max(highest, -lowest) >= _FLOAT64_EXACT_LIMIT:
        suspect |= np.abs(values) >= _FLOAT64_EXACT_LIMIT
    rows = np.flatnonzero(suspect)
    for token, number in zip(text.take(pa.array(rows)).to_pylist(), values[rows].tolist(), strict=True):
        exact = Decimal(token)
        if number == 0 and exact != 0:
            found = f"{name!r} value {token!r} underflowing to zero"
        elif exact == exact.to_integral_value() and exact != Decimal(number):
            found = f"{name!r} value {token!r} beyond the exact float64 integers"
        else:
            continue
        raise ReaderFormatError(context=context, path=path, found=found, expected=expected)
    return values


def decode_integers(
    text: pa.Array | pa.ChunkedArray,
    /,
    *,
    name: str,
    minimum: int | None,
    context: ReaderErrorContext,
    path: Path,
) -> npt.NDArray[np.int64 | np.uint64]:
    """Decode text into exact 64-bit integers without a floating intermediate.

    Parameters
    ----------
    text
        Arrow string values, one per row (other string types are cast to
        ``string``).
    name
        Column name used in error messages.
    minimum
        Smallest accepted value. ``None`` accepts every int64 value. With a
        nonnegative minimum, values up to ``2**64 - 1`` are accepted;
        otherwise values must fit int64.
    context
        Reader and artifact identity for invalid values.
    path
        Artifact that stores the values, for invalid values.

    Returns
    -------
    numpy.ndarray
        One exact value per row: int64, or uint64 when a value is at least
        2**63. It is a read-only view of Arrow memory when ``text`` is one
        array of plain decimal integers.

    Raises
    ------
    ReaderFormatError
        If a value is missing, not an exact decimal spelling of an integer, or
        outside ``[minimum, 2**64)`` (int64 for a negative or no minimum).
    """
    lower = -(2**63) if minimum is None else minimum
    upper = 2**64 if lower >= 0 else 2**63
    expected = f"exact integer {name!r} values" + ("" if minimum is None else f" of at least {minimum}")
    expected += " below 2**64" if lower >= 0 else " within int64"

    if text.null_count:
        raise ReaderFormatError(context=context, path=path, found=f"missing {name!r} values", expected=expected)
    if text.type != pa.string():
        text = pc.cast(text, pa.string())
    values = _cast_plain_integers(text, unsigned=lower >= 0)
    if values is not None:
        if not len(values) or values.min() >= lower:
            return values
        token = text[int(values.argmin())].as_py()
    else:
        integers: list[int] = []
        for token in text.to_pylist():
            integer = _exact_integer(token, lower, upper)
            if integer is None:
                break
            integers.append(integer)
        else:  # Every token is an exact integer in range.
            return np.asarray(integers, dtype=np.uint64 if integers and max(integers) >= 2**63 else np.int64)
    raise ReaderFormatError(context=context, path=path, found=f"{name!r} value {token!r}", expected=expected)


def decode_integer_block(
    columns: Sequence[pa.Array],
    /,
    *,
    names: Sequence[str],
    minimum: int | None,
    context: ReaderErrorContext,
    path: Path,
) -> npt.NDArray[np.int64 | np.uint64]:
    """Decode equally long text columns at once with :func:`decode_integers`.

    Parameters
    ----------
    columns
        Arrow string arrays of one bounded block, one per column.
    names
        Column names in ``columns`` order, used in error messages.
    minimum
        Smallest accepted value, as for :func:`decode_integers`.
    context
        Reader and artifact identity for invalid values.
    path
        Artifact that stores the values, for invalid values.

    Returns
    -------
    numpy.ndarray
        Rows-by-columns values: int64, or uint64 when a value is at least
        2**63. It is a transposed view of column-major values.

    Raises
    ------
    ReaderFormatError
        As :func:`decode_integers`, naming the column of the invalid value.
    """
    return _decode_block(
        columns,
        names,
        lambda text, name: decode_integers(text, name=name, minimum=minimum, context=context, path=path),
    )


def decode_float64_block(
    columns: Sequence[pa.Array], /, *, names: Sequence[str], context: ReaderErrorContext, path: Path
) -> npt.NDArray[np.float64]:
    """Decode equally long text columns at once with :func:`decode_float64`.

    Parameters
    ----------
    columns
        Arrow string arrays of one bounded block, one per column.
    names
        Column names in ``columns`` order, used in error messages.
    context
        Reader and artifact identity for invalid values.
    path
        Artifact that stores the values, for invalid values.

    Returns
    -------
    numpy.ndarray
        Rows-by-columns float64 values, a transposed view of column-major
        values.

    Raises
    ------
    ReaderFormatError
        As :func:`decode_float64`, naming the column of the invalid value.
    """
    return _decode_block(columns, names, lambda text, name: decode_float64(text, name=name, context=context, path=path))


def _decode_block[T: np.generic](
    columns: Sequence[pa.Array], names: Sequence[str], decode: Callable[[pa.Array, str], npt.NDArray[T]]
) -> npt.NDArray[T]:
    """Decode concatenated columns once; for an invalid block, raise the error of the first invalid column."""
    try:
        values = decode(pa.concat_arrays(list(columns)), "block")
    except ReaderFormatError as error:
        # A value that is invalid in the block is invalid in its own column too.
        failures = (_failure(decode, column, name) for column, name in zip(columns, names, strict=True))
        raise next((failure for failure in failures if failure is not None), error) from None
    return values.reshape(len(columns), -1).T


def _failure[T: np.generic](
    decode: Callable[[pa.Array, str], npt.NDArray[T]], column: pa.Array, name: str
) -> ReaderFormatError | None:
    """Return the error of decoding one column, or None for a valid column."""
    try:
        decode(column, name)
    except ReaderFormatError as error:
        return error
    return None


def _cast_plain_integers(
    text: pa.Array | pa.ChunkedArray, *, unsigned: bool
) -> npt.NDArray[np.int64 | np.uint64] | None:
    """Cast optionally negative ASCII digit strings with Arrow; return None for any other text."""
    # Arrow's integer cast also accepts hexadecimal such as 0x10, so only text
    # of digits and minus signs may take it; Arrow rejects a misplaced sign.
    if not _digits_and_minus_only(text):
        return None
    try:
        return pc.cast(text, pa.int64()).to_numpy(zero_copy_only=False)
    except pa.ArrowInvalid:
        # A plain integer outside int64; uint64 holds it if none is negative.
        if not unsigned:
            return None
    try:
        return pc.cast(text, pa.uint64()).to_numpy(zero_copy_only=False)
    except pa.ArrowInvalid:
        return None


def _digits_and_minus_only(text: pa.Array | pa.ChunkedArray) -> bool:
    """Return whether every character of utf8 ``text`` is an ASCII digit or ``-``."""
    # One vectorized pass over each chunk's character buffer is about ten times
    # faster than a per-value string kernel on short numeric fields.
    for chunk in text.chunks if isinstance(text, pa.ChunkedArray) else (text,):
        offsets, characters = chunk.buffers()[1:]
        start, stop = np.frombuffer(offsets, np.int32)[[chunk.offset, chunk.offset + len(chunk)]]
        if stop > start:
            codes = np.frombuffer(characters, np.uint8, count=stop - start, offset=start)
            if not (((codes >= ord("0")) & (codes <= ord("9"))) | (codes == ord("-"))).all():
                return False
    return True


def _exact_integer(token: str, lower: int, upper: int) -> int | None:
    """Return the integer that a decimal spelling denotes within ``[lower, upper)``, else None."""
    if not _DECIMAL.fullmatch(token):
        return None
    exact = Decimal(token)
    # Checking the range first keeps huge exponents away from int().
    if not lower <= exact < upper or exact != exact.to_integral_value():
        return None
    return int(exact)


def _full_match(text: pa.Array | pa.ChunkedArray, pattern: str) -> npt.NDArray[np.bool_]:
    """Return whether each value matches the RE2 ``pattern`` as a whole."""
    options = pc.MatchSubstringOptions(f"^(?:{pattern})$")
    return pc.call_function("match_substring_regex", [text], options).to_numpy(zero_copy_only=False)
