from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence


def _normalize_axes(axes: Sequence[str], *, parameter: str) -> tuple[str, ...]:
    normalized = tuple(axis.lower() for axis in axes)
    if any(len(axis) != 1 or not axis.isalpha() for axis in normalized):
        message = f"{parameter} must contain single alphabetic axis names; found {tuple(axes)!r}."
        raise ValueError(message)
    if len(set(normalized)) != len(normalized):
        message = f"{parameter} contains duplicate axes: {tuple(axes)!r}."
        raise ValueError(message)
    return normalized


def _validate_axes_contract(data: da.Array, source: tuple[str, ...], target: tuple[str, ...]) -> None:
    if len(source) != data.ndim:
        message = (
            f"source_axes rank {len(source)} does not match array rank {data.ndim}; "
            f"shape={data.shape}, axes={source!r}."
        )
        raise ValueError(message)
    if not {"x", "y"}.issubset(source) or not {"x", "y"}.issubset(target):
        message = f"Raster source and target axes must both contain 'x' and 'y'; source={source}, target={target}."
        raise ValueError(message)


def _normalize_selectors(
    selectors: Mapping[str, int] | None,
    source: tuple[str, ...],
    target: tuple[str, ...],
) -> dict[str, int]:
    normalized = {axis.lower(): index for axis, index in (selectors or {}).items()}
    if len(normalized) != len(selectors or {}):
        message = "selectors contains duplicate axes after case normalization."
        raise ValueError(message)
    unknown = set(normalized).difference(source)
    if unknown:
        message = f"selectors contains axes absent from source_axes: {sorted(unknown)}."
        raise ValueError(message)
    selected_targets = set(normalized).intersection(target)
    if selected_targets:
        message = f"Axes named in selectors must be omitted from target_axes; found {sorted(selected_targets)}."
        raise ValueError(message)
    return normalized


def _select_source_axes(
    data: da.Array,
    source: tuple[str, ...],
    target: tuple[str, ...],
    selectors: Mapping[str, int],
) -> tuple[da.Array, list[str]]:
    indexer: list[slice | int] = []
    remaining: list[str] = []
    for axis, size in zip(source, data.shape, strict=True):
        if axis in target:
            indexer.append(slice(None))
            remaining.append(axis)
        elif axis in selectors:
            index = selectors[axis]
            if index < 0 or index >= size:
                message = f"Selector for axis {axis!r} is {index}, but its valid range is [0, {size})."
                raise ValueError(message)
            indexer.append(index)
        elif size == 1:
            indexer.append(0)
        else:
            message = f"Axis {axis!r} has length {size} and is absent from target_axes; provide an explicit selector."
            raise ValueError(message)
    return data[tuple(indexer)], remaining


def _add_target_axes(
    data: da.Array,
    remaining: list[str],
    target: tuple[str, ...],
    add_missing: Collection[str],
) -> tuple[da.Array, list[str]]:
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
    selectors: Mapping[str, int] | None = None,
    add_missing: Collection[str] = (),
) -> da.Array:
    """Select, add, and transpose explicitly named raster axes without computing pixels.

    Names are case-insensitive. Unwanted non-singleton axes require explicit
    integer selectors; unwanted singleton axes may be dropped. Missing target
    axes are added only when named in ``add_missing``. Both source and target
    must contain the spatial axes ``x`` and ``y``.

    Parameters
    ----------
    data
        Lazy source array. Its rank must equal the number of ``source_axes``.
    source_axes
        File-declared axes in current array order.
    target_axes
        Required axes and their returned order.
    selectors
        Integer selections for source axes intentionally omitted from the
        target. A selector can never silently select a retained axis.
    add_missing
        Target axes that may be inserted as length-one dimensions.

    Returns
    -------
    dask.array.Array
        A lazy view built only from slicing, singleton insertion, and
        transposition.

    Raises
    ------
    ValueError
        If axes are invalid, ambiguous, incomplete, or require an implicit
        non-singleton selection.
    """
    source = _normalize_axes(source_axes, parameter="source_axes")
    target = _normalize_axes(target_axes, parameter="target_axes")
    _validate_axes_contract(data, source, target)
    normalized_selectors = _normalize_selectors(selectors, source, target)
    result, remaining = _select_source_axes(data, source, target, normalized_selectors)
    result, remaining = _add_target_axes(result, remaining, target, add_missing)
    return result.transpose(tuple(remaining.index(axis) for axis in target))
