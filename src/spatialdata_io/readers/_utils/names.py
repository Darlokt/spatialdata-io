"""SpatialData element-name and table column-name rules shared by readers.

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

Table column names taken from file headers follow the second policy:
:func:`sanitize_column_names` repairs the names of one ``obs`` or ``var``
frame, or the top-level keys of ``uns``. SpatialData's ``TableModel`` and
``SpatialData.write`` call ``validate_table_attr_keys`` in
``spatialdata._core.validation`` (spatialdata 0.8.0). It checks every ``obs``
and ``var`` column name with ``check_valid_dataframe_column_name`` (the
element-name rule above, and not ``_index``), the top-level keys of ``uns``,
``obsm``, ``obsp``, ``varm``, ``varp`` and ``layers`` with the element-name
rule only (nested ``uns`` keys are not checked), and requires the names of
each of these to be unique ignoring letter case. Points and shapes column
names are not checked.
"""

from __future__ import annotations

import os
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

_RULE = "nonempty, only alphanumeric characters, '_', '-' or '.', not '.' or '..', and not starting with '__'"
_FALLBACK = "unnamed"
_INDEX_COLUMN = "_index"
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


def _repair(value: str) -> str:
    """Apply the character, leading-underscore and empty-name repairs to ``value``."""
    name = "".join(character if character.isalnum() or character in "_-." else "_" for character in value)
    name = _LEADING_UNDERSCORES.sub("_", name)
    return _FALLBACK if name in {"", ".", ".."} else name


def _remedy(parameter: str | None) -> str:
    """Return the warning sentence telling the user how to choose a name."""
    if parameter is None:
        return "Rename the source files to choose the name explicitly."
    return f"Pass {parameter} to choose the name explicitly."


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


def sanitize_element_name(value: str, /, *, source: str, parameter: str | None, reserved: Collection[str] = ()) -> str:
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
        Caller option that sets the name explicitly, named in the warning, or
        ``None`` when the reader has no such option; the warning then suggests
        renaming the source files.
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
        Once, naming ``value``, the repaired name and ``parameter`` (or the
        renaming remedy), if the name changed. The warning points at the first
        caller outside ``spatialdata_io``.
    """
    reserved_names = {reserved_name.lower() for reserved_name in reserved}
    if _follows_rule(value) and value.lower() not in reserved_names:
        return value

    name = _repair(value)
    while name.lower() in reserved_names:
        name = _FALLBACK if name == "_" else f"{name}_"

    warnings.warn(
        f"{source} {value!r} is not a usable SpatialData element name; using {name!r}. {_remedy(parameter)}",
        UserWarning,
        skip_file_prefixes=(_PACKAGE_PREFIX,),
    )
    return name


def sanitize_column_names(
    columns: Sequence[str], /, *, source: str, reserved: Collection[str] = (), kind: Literal["column", "uns"] = "column"
) -> list[str]:
    """Return usable SpatialData table column names for one frame, warning if any had to change.

    A name is kept unchanged when it follows the element-name rule, is not
    ``_index`` (columns only) and does not equal, ignoring letter case, a
    reserved name or an earlier kept name. The others are repaired
    deterministically, in column order, as ``spatialdata.sanitize_table`` does:

    1. the character, leading-underscore and empty-name repairs of
       :func:`sanitize_element_name` (``"CD45 RO"`` -> ``"CD45_RO"``), and
       a column ``_index`` becomes ``index``;
    2. while the result equals a name already taken ignoring letter case,
       the first free ``_1``, ``_2``, ... suffix is appended
       (``"CD45 RO"`` next to ``"CD45_RO"`` -> ``"CD45_RO_1"``); a taken
       ``_`` becomes ``_1``, ``_2``, ... because ``__`` is invalid.

    Kept names are taken before any repaired name, so a repair never renames
    a valid source column. The result is unique ignoring letter case,
    including against ``reserved``, and passes SpatialData's column-name
    check. The repair is idempotent: repairing the result changes nothing.

    Parameters
    ----------
    columns
        Distinct column names of one ``obs`` or ``var`` frame, in order,
        usually from a file header. Readers reject duplicate header names
        under their own policy before calling this.
    source
        Description of where the names come from, starting the warning, for
        example ``"mcmicro: the quantification CSV header"``.
    reserved
        Reader-owned columns of the same frame, which keep their spelling.
        Another column equal to one of them ignoring letter case, but not
        exactly, is renamed; an exact match is the reserved column itself.
    kind
        ``"column"`` for ``obs`` or ``var`` column names; ``"uns"`` for
        top-level ``uns`` keys, which SpatialData checks with the element-name
        rule only, so ``_index`` is kept. The warning names the kind.

    Returns
    -------
    list[str]
        The usable names, in the order of ``columns``.

    Raises
    ------
    ValueError
        If ``columns`` contains a name twice.

    Warns
    -----
    UserWarning
        Once, listing every changed name as ``original -> repaired``. The
        warning points at the first caller outside ``spatialdata_io``.
    """
    counts = Counter(columns)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        message = f"{source} has duplicate {'column names' if kind == 'column' else 'names'} {duplicates!r}."
        raise ValueError(message)

    # SpatialData forbids "_index" only as an obs or var column name.
    forbidden = _INDEX_COLUMN if kind == "column" else None
    taken = {name.lower() for name in reserved}
    reserved_names = frozenset(reserved)
    kept: list[bool] = []
    for name in columns:
        keep = name in reserved_names or (_follows_rule(name) and name != forbidden and name.lower() not in taken)
        if keep:
            taken.add(name.lower())
        kept.append(keep)

    result: list[str] = []
    renamed: list[tuple[str, str]] = []
    for name, keep in zip(columns, kept, strict=True):
        if keep:
            result.append(name)
            continue
        repaired = _repair(name)
        candidate = "index" if repaired == forbidden else repaired
        # "_" plus a suffix would start with "__", so "_" is numbered "_1", "_2", ...
        stem, counter = ("" if candidate == "_" else candidate), 1
        while candidate.lower() in taken:
            candidate, counter = f"{stem}_{counter}", counter + 1
        taken.add(candidate.lower())
        result.append(candidate)
        renamed.append((name, candidate))

    if renamed:
        pairs = ", ".join(f"{name!r} -> {candidate!r}" for name, candidate in renamed)
        problem, remedy = (
            ("column names that a SpatialData table does not accept", "Rename the source columns")
            if kind == "column"
            else ("names that a SpatialData table does not accept as uns keys", "Rename them in the source")
        )
        warnings.warn(
            f"{source} has {problem}; using {pairs}. {remedy} to choose the names explicitly.",
            UserWarning,
            skip_file_prefixes=(_PACKAGE_PREFIX,),
        )
    return result
