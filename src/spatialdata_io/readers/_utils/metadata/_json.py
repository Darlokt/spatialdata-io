"""Implement strict JSON-object parsing for private reader layouts.

The module owns only the strict JSON contract: UTF-8 text, a top-level object,
unique object names and numbers that are finite as floats. Readers retain
artifact discovery, schema validation, version semantics, and interpretation
of metadata values.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Never, cast

from spatialdata_io.readers._utils.errors import (
    ReaderErrorContext,
    ReaderFormatError,
    add_reader_context,
)

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.paths import ArtifactFile

_EXPECTED = "a UTF-8 JSON object with unique names and finite numbers"
_DUPLICATE_NAME = "a duplicate object name"
_NUMBER_OVERFLOW = "a number outside the finite float range"
_NONSTANDARD_NUMBER = "a nonstandard non-finite numeric constant"
_JSON_VALUE_KINDS: dict[type[object], str] = {
    type(None): "null",
    bool: "boolean",
    int: "number",
    float: "number",
    str: "string",
    list: "array",
}


class _JsonContentError(ValueError):
    """Strict-JSON violation raised by a decoder hook and reported as ``ReaderFormatError``."""

    found: str

    def __init__(self, found: str) -> None:
        """Store the description of the offending content."""
        self.found = found
        super().__init__(found)


def _plain_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a plain dictionary from object pairs, rejecting duplicate names."""
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise _JsonContentError(_DUPLICATE_NAME)
        result[name] = value
    return result


def _finite_float(value: str) -> float:
    """Parse a JSON float, rejecting a literal that overflows to infinity."""
    result = float(value)
    if not math.isfinite(result):
        raise _JsonContentError(_NUMBER_OVERFLOW)
    return result


def _finite_integer(value: str) -> int:
    """Parse a JSON integer, rejecting one outside the finite float range."""
    # Readers compare and convert numbers as floats; an integer beyond the
    # float range would otherwise surface later as a bare OverflowError.
    result = int(value)
    try:
        float(result)
    except OverflowError:
        raise _JsonContentError(_NUMBER_OVERFLOW) from None
    return result


def _reject_nonstandard_constant(_value: str) -> Never:
    """Reject the ``NaN`` and ``Infinity`` literals that strict JSON does not define."""
    raise _JsonContentError(_NONSTANDARD_NUMBER)


def parse_json_object(
    text: str,
    /,
    *,
    context: ReaderErrorContext,
    path: Path,
) -> dict[str, object]:
    """Parse JSON text as an owned JSON object.

    Use this for JSON embedded in another container, such as an HDF5
    attribute; use :func:`read_json_object` for a JSON file.

    Parameters
    ----------
    text
        Decoded JSON text.
    context
        Reader, artifact, and optional detected-version identity included in
        content failures.
    path
        Artifact that stores the text, for content failures.

    Returns
    -------
    dict[str, object]
        A newly owned plain dictionary. Nested JSON objects are plain
        dictionaries as well.

    Raises
    ------
    ReaderFormatError
        If the text is not strict JSON, contains duplicate object names, a
        non-finite number or an integer beyond the float range, exceeds parser
        limits, or does not have a top-level object.

    Notes
    -----
    The function does not validate or coerce vendor-specific keys and values.
    """
    try:
        parsed: object = json.loads(
            text,
            object_pairs_hook=_plain_object,
            parse_constant=_reject_nonstandard_constant,
            parse_float=_finite_float,
            parse_int=_finite_integer,
        )
    except _JsonContentError as error:
        raise ReaderFormatError(context=context, found=error.found, expected=_EXPECTED, path=path) from error
    except json.JSONDecodeError as error:
        raise ReaderFormatError(
            context=context,
            found=f"malformed JSON at line {error.lineno}, column {error.colno}",
            expected=_EXPECTED,
            path=path,
        ) from error
    except RecursionError as error:
        raise ReaderFormatError(
            context=context, found="JSON nesting beyond the parser limit", expected=_EXPECTED, path=path
        ) from error
    except ValueError as error:
        # Python's integer-string conversion limit.
        raise ReaderFormatError(
            context=context, found="a JSON numeric value beyond the parser limit", expected=_EXPECTED, path=path
        ) from error

    if type(parsed) is not dict:
        raise ReaderFormatError(
            context=context,
            found=f"a top-level JSON {_JSON_VALUE_KINDS[type(parsed)]}",
            expected=_EXPECTED,
            path=path,
        )

    # ``object_pairs_hook`` constructs every JSON object with this exact type;
    # typeshed necessarily types the dynamic decoder result as ``Any``.
    return cast("dict[str, object]", parsed)


def read_json_object(
    path: ArtifactFile,
    /,
    *,
    context: ReaderErrorContext,
) -> dict[str, object]:
    """Read one validated regular file as an owned JSON object.

    Parameters
    ----------
    path
        Canonical regular file previously validated by the shared path
        boundary. The path is borrowed and is not resolved or kind-checked
        again.
    context
        Reader, artifact, and optional detected-version identity included in
        content failures and added to I/O failures.

    Returns
    -------
    dict[str, object]
        A newly owned plain dictionary independent of the closed source file.
        Nested JSON objects are plain dictionaries as well.

    Raises
    ------
    ReaderFormatError
        If the file is not strict UTF-8 JSON or violates the contract of
        :func:`parse_json_object`.
    OSError
        If opening, reading, or closing the file fails. The original exception
        is preserved with reader context.

    Notes
    -----
    Parsing is eager because reader layout and version decisions need the
    metadata immediately. The file is decoded once as strict UTF-8 text, as
    the standard decoder would read it. No reader-selected file-size ceiling
    is imposed.
    """
    try:
        with path.open(encoding="utf-8", errors="strict") as source:
            text = source.read()
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise
    except UnicodeDecodeError as error:
        raise ReaderFormatError(
            context=context,
            found="content that is not valid UTF-8",
            expected=_EXPECTED,
            path=path,
        ) from error
    return parse_json_object(text, context=context, path=path)
