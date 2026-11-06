"""Select, insert and reorder explicitly named raster axes of a lazy array."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence


def _normalize_axes(axes: Sequence[str], *, parameter: str) -> tuple[str, ...]:
    """Return lowercased axis names, requiring unique single alphabetic letters."""
    normalized = tuple(axis.lower() for axis in axes)
    if any(len(axis) != 1 or not axis.isalpha() for axis in normalized):
        message = f"{parameter} must contain single alphabetic axis names; found {tuple(axes)!r}."
        raise ValueError(message)
    if len(set(normalized)) != len(normalized):
        message = f"{parameter} contains duplicate axes: {tuple(axes)!r}."
        raise ValueError(message)
    return normalized


def _validate_axes_contract(data: da.Array, source: tuple[str, ...], target: tuple[str, ...]) -> None:
    """Require the source rank to match ``data`` and both axis sets to contain ``x`` and ``y``."""
    if len(source) != data.ndim:
        message = (
            f"source_axes rank {len(source)} does not match array rank {data.ndim}; "
            f"shape={data.shape}, axes={source!r}."
        )
        raise ValueError(message)
    if not {"x", "y"}.issubset(source) or not {"x", "y"}.issubset(target):
        message = f"Raster source and target axes must both contain 'x' and 'y'; source={source}, target={target}."
        raise ValueError(message)


def _drop_source_axes(data: da.Array, source: tuple[str, ...], target: tuple[str, ...]) -> tuple[da.Array, list[str]]:
    """Drop singleton source axes absent from the target; any other omission is ambiguous."""
    indexer: list[slice | int] = []
    remaining: list[str] = []
    for axis, size in zip(source, data.shape, strict=True):
        if axis in target:
            indexer.append(slice(None))
            remaining.append(axis)
        elif size == 1:
            indexer.append(0)
        else:
            message = f"Axis {axis!r} has length {size} and is absent from target_axes."
            raise ValueError(message)
    return data[tuple(indexer)], remaining


def _add_target_axes(
    data: da.Array,
    remaining: list[str],
    target: tuple[str, ...],
    add_missing: Collection[str],
) -> tuple[da.Array, list[str]]:
    """Append length-one dimensions for target axes named in ``add_missing``.

    Return the expanded array and the updated axis order; any other absent target axis is an error.
    """
    allowed = set(_normalize_axes(tuple(add_missing), parameter="add_missing"))
    invalid = allowed.difference(target)
    if invalid:
        message = f"add_missing contains axes absent from target_axes: {sorted(invalid)}."
        raise ValueError(message)
    for axis in target:
        if axis in remaining:
            continue
        if axis not in allowed:
            message = f"Target axis {axis!r} is absent from source_axes and was not listed in add_missing."
            raise ValueError(message)
        data = da.expand_dims(data, axis=data.ndim)
        remaining.append(axis)
    return data, remaining


def normalize_raster_axes(
    data: da.Array,
    source_axes: Sequence[str],
    *,
    target_axes: Sequence[str],
    add_missing: Collection[str] = (),
) -> da.Array:
    """Drop, add, and transpose explicitly named raster axes without computing pixels.

    Names are case-insensitive single letters. Readers may use a format's
    declared axis labels or an explicit reader-owned override; this helper does
    not assign semantics to unfamiliar labels. Singleton source axes absent
    from the target are dropped; a non-singleton one is an error, so no plane
    is ever selected implicitly. Missing target axes are added only when named
    in ``add_missing``. Both source and target must contain the spatial axes
    ``x`` and ``y``.

    Parameters
    ----------
    data
        Lazy source array. Its rank must equal the number of ``source_axes``.
    source_axes
        File-declared axes in current array order.
    target_axes
        Required axes and their returned order.
    add_missing
        Target axes that may be inserted as length-one dimensions.

    Returns
    -------
    dask.array.Array
        A lazy view built only from singleton removal and insertion, and
        transposition.

    Raises
    ------
    ValueError
        If axes are invalid, ambiguous or incomplete, or a non-singleton source
        axis is absent from the target.
    """
    source = _normalize_axes(source_axes, parameter="source_axes")
    target = _normalize_axes(target_axes, parameter="target_axes")
    _validate_axes_contract(data, source, target)
    result, remaining = _drop_source_axes(data, source, target)
    result, remaining = _add_target_axes(result, remaining, target, add_missing)
    return result.transpose(tuple(remaining.index(axis) for axis in target))
