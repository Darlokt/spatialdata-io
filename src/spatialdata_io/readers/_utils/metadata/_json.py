"""Implement bounded JSON-object reading for private reader layouts.

The module owns only the regular-file JSON storage contract. Readers retain
artifact discovery, byte-limit selection, schema validation, version semantics,
and interpretation of metadata values.
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
    from spatialdata_io.readers._utils.paths import ArtifactFile

_DUPLICATE_NAME = "a duplicate object name"
_FLOAT_OVERFLOW = "a number outside the finite float range"
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
    found: str

    def __init__(self, found: str) -> None:
        self.found = found
        super().__init__(found)


def _plain_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise _JsonContentError(_DUPLICATE_NAME)
        result[name] = value
    return result


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise _JsonContentError(_FLOAT_OVERFLOW)
    return result


def _reject_nonstandard_constant(_value: str) -> Never:
    raise _JsonContentError(_NONSTANDARD_NUMBER)


def _expected_json_object(max_bytes: int) -> str:
    return f"a UTF-8 JSON object with unique names and finite numbers no larger than {max_bytes} bytes"


def _json_value_kind(value: object) -> str:
    return _JSON_VALUE_KINDS[type(value)]


def read_json_object(
    path: ArtifactFile,
    /,
    *,
    max_bytes: int,
    context: ReaderErrorContext,
) -> dict[str, object]:
    """Read one validated regular file as a bounded owned JSON object.

    Parameters
    ----------
    path
        Canonical regular file previously validated by the shared path
        boundary. The path is borrowed and is not resolved or kind-checked
        again.
    max_bytes
        Positive reader-selected maximum encoded size. At most one additional
        byte is read to detect oversized content.
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
    ValueError
        If ``max_bytes`` is not positive.
    ReaderFormatError
        If the file is oversized, is not strict UTF-8 JSON, contains duplicate
        object names or non-finite numbers, exceeds parser limits, or does not
        have a top-level object.
    OSError
        If opening, reading, or closing the file fails. The original exception
        is preserved with reader context.

    Notes
    -----
    Parsing is eager because reader layout and version decisions need this
    bounded metadata immediately. The function performs one bounded binary
    read, one strict UTF-8 decode, and one JSON parse. It does not validate or
    coerce vendor-specific keys and values.
    """
    if max_bytes <= 0:
        message = "max_bytes must be positive."
        raise ValueError(message)

    try:
        with path.open("rb") as file:
            payload = file.read(max_bytes + 1)
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise

    expected = _expected_json_object(max_bytes)
    if len(payload) > max_bytes:
        raise ReaderFormatError(
            context=context,
            found=f"more than {max_bytes} encoded bytes",
            expected=expected,
            path=path,
        )

    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ReaderFormatError(
            context=context,
            found="content that is not valid UTF-8",
            expected=expected,
            path=path,
        ) from error

    try:
        parsed: object = json.loads(
            text,
            object_pairs_hook=_plain_object,
            parse_constant=_reject_nonstandard_constant,
            parse_float=_finite_float,
        )
    except _JsonContentError as error:
        raise ReaderFormatError(
            context=context,
            found=error.found,
            expected=expected,
            path=path,
        ) from error
    except json.JSONDecodeError as error:
        raise ReaderFormatError(
            context=context,
            found=f"malformed JSON at line {error.lineno}, column {error.colno}",
            expected=expected,
            path=path,
        ) from error
    except RecursionError as error:
        raise ReaderFormatError(
            context=context,
            found="JSON nesting beyond the parser limit",
            expected=expected,
            path=path,
        ) from error
    except ValueError as error:
        raise ReaderFormatError(
            context=context,
            found="a JSON numeric value beyond the parser limit",
            expected=expected,
            path=path,
        ) from error

    if type(parsed) is not dict:
        raise ReaderFormatError(
            context=context,
            found=f"a top-level JSON {_json_value_kind(parsed)}",
            expected=expected,
            path=path,
        )

    # ``object_pairs_hook`` constructs every JSON object with this exact type;
    # typeshed necessarily types the dynamic decoder result as ``Any``.
    return cast("dict[str, object]", parsed)
