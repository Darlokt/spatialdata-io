"""SpatialData element-name validation shared by readers.

Readers turn caller options (``dataset_id``) and source values (sample names,
region names, file stems) into SpatialData element and coordinate-system names.
:func:`validate_element_name` checks such a name once, before it is used, with
SpatialData's own rule, so that an unusable name fails with the reader's
parameter name instead of later inside :class:`spatialdata.SpatialData` or on
write.

The rule mirrors ``spatialdata._core.validation.check_valid_name`` in
spatialdata 0.8.0, which SpatialData calls when elements are added but does not
export. A valid name is

- a nonempty ``str``,
- not ``.`` or ``..``,
- not starting with ``__``, and
- made only of characters for which :meth:`str.isalnum` is true (any Unicode
  letter or digit) and ``_``, ``-`` or ``.``.

SpatialData also requires element names to be unique ignoring letter case
(``check_all_keys_case_insensitively_unique``, which compares
:meth:`str.lower`), so a reader-reserved name such as ``table`` also excludes
``Table``. Which names are reserved, and which further collisions matter, stays
with the reader that owns the element names.

Names come from two kinds of source, with different policies:

- explicit caller values (``dataset_id=...``) are checked with
  :func:`validate_element_name` and fail when unusable, because the caller can
  correct them;
- names derived from files (a filename stem, a directory name) are repaired
  with :func:`sanitize_element_name`, which warns once with the original and
  the repaired name, so a dataset named ``my run.h5ad`` still reads.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection

_RULE = "nonempty, only alphanumeric characters, '_', '-' or '.', not '.' or '..', and not starting with '__'"
_FALLBACK = "unnamed"
_LEADING_UNDERSCORES = re.compile(r"^_{2,}")
# Warnings name the first frame outside this package, the user's call site,
# however deeply a reader calls the sanitizer.
_PACKAGE_PREFIX = f"{Path(__file__).parents[2]}{os.sep}"


def _follows_rule(name: str) -> bool:
    """Return whether ``name`` satisfies SpatialData's element-name rule."""
    return (
        bool(name)
        and name not in {".", ".."}
        and not name.startswith("__")
        and all(character.isalnum() or character in "_-." for character in name)
    )


def validate_element_name(name: object, /, *, parameter: str, reserved: Collection[str] = ()) -> str:
    """Return ``name`` after checking it is a usable SpatialData element name.

    Parameters
    ----------
    name
        Untrusted candidate name, from a caller option or a source artifact.
    parameter
        Description of where the name comes from, such as ``"dataset_id"``.
        Error messages start with it.
    reserved
        Names the reader already uses for other elements. A candidate equal to
        one of them ignoring letter case is rejected, as SpatialData would
        reject the collision.

    Returns
    -------
    str
        ``name`` unchanged.

    Raises
    ------
    TypeError
        If ``name`` is not a ``str``.
    ValueError
        If ``name`` violates SpatialData's element-name rule (see the module
        docstring) or equals a reserved name ignoring letter case. The message
        names ``parameter`` and the rejected value.

    Notes
    -----
    Readers that take the name from a file rather than from a caller option
    convert the ``ValueError`` into a ``ReaderFormatError`` with the artifact's
    context.
    """
    if not isinstance(name, str):
        message = f"{parameter} must be a string; found {type(name).__name__}."
        raise TypeError(message)

    if not _follows_rule(name):
        message = f"{parameter} must be a valid SpatialData element name: {_RULE}; found {name!r}."
        raise ValueError(message)

    if name.lower() in {reserved_name.lower() for reserved_name in reserved}:
        message = (
            f"{parameter} must be a valid SpatialData element name other than {sorted(reserved)!r} in any letter "
            f"case, because SpatialData compares element names case-insensitively; found {name!r}."
        )
        raise ValueError(message)

    return name


def sanitize_element_name(value: str, /, *, source: str, parameter: str, reserved: Collection[str] = ()) -> str:
    """Return a valid element name derived from a file, warning if ``value`` had to change.

    The repair is deterministic and idempotent, and a valid name that is not
    reserved is returned unchanged without a warning. Otherwise, in order:

    1. every character that is not alphanumeric, ``_``, ``-`` or ``.`` becomes
       ``_`` (``"my run"`` -> ``"my_run"``);
    2. a leading run of two or more underscores becomes one (``"__x"`` ->
       ``"_x"``, ``"__"`` -> ``"_"``);
    3. an empty value, ``.`` and ``..`` become ``"unnamed"``;
    4. while the name equals a reserved name ignoring letter case, ``_`` is
       appended (``"Table"`` -> ``"Table_"``); a reserved ``"_"`` becomes
       ``"unnamed"`` first, because ``"__"`` would be invalid.

    The result always passes :func:`validate_element_name` with the same
    ``reserved``. Different values can map to the same name; collisions between
    several derived names stay with the reader.

    Parameters
    ----------
    value
        Name taken from a source artifact, such as a filename stem.
    source
        Description of where ``value`` comes from, starting the warning, for
        example ``"dbit: the H5AD filename stem"``.
    parameter
        Caller option that sets the name explicitly, named in the warning.
    reserved
        Names the reader already uses for other elements, compared ignoring
        letter case.

    Returns
    -------
    str
        ``value`` itself if it is usable, otherwise the repaired name.

    Warns
    -----
    UserWarning
        Once, naming ``value``, the repaired name and ``parameter``, if the
        name changed. The warning points at the first caller outside
        ``spatialdata_io``.
    """
    reserved_names = {reserved_name.lower() for reserved_name in reserved}
    if _follows_rule(value) and value.lower() not in reserved_names:
        return value

    name = "".join(character if character.isalnum() or character in "_-." else "_" for character in value)
    name = _LEADING_UNDERSCORES.sub("_", name)
    if name in {"", ".", ".."}:
        name = _FALLBACK
    while name.lower() in reserved_names:
        name = _FALLBACK if name == "_" else f"{name}_"

    warnings.warn(
        f"{source} {value!r} is not a usable SpatialData element name; using {name!r}. "
        f"Pass {parameter} to choose the name explicitly.",
        UserWarning,
        skip_file_prefixes=(_PACKAGE_PREFIX,),
    )
    return name
