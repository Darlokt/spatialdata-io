"""Classify invalid source polygons and report them with a shared contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import shapely

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

type InvalidGeometryPolicy = Literal["keep", "drop", "raise"]

_POLICIES: tuple[InvalidGeometryPolicy, ...] = ("keep", "drop", "raise")
_EXAMPLES = 10


def validate_invalid_geometry_policy(value: object, /) -> InvalidGeometryPolicy:
    """Return a validated public ``invalid_geometries`` option.

    Parameters
    ----------
    value
        Caller-supplied option value.

    Returns
    -------
    InvalidGeometryPolicy
        ``"keep"`` preserves source geometry, ``"drop"`` removes reader-defined
        entities with invalid geometry and ``"raise"`` rejects them.

    Raises
    ------
    TypeError
        If ``value`` is not a string.
    ValueError
        If ``value`` is not one of the three policies.
    """
    if not isinstance(value, str):
        message = f"invalid_geometries must be one of {_POLICIES}; found {type(value).__name__}."
        raise TypeError(message)
    for policy in _POLICIES:
        if value == policy:
            return policy
    message = f"invalid_geometries must be one of {_POLICIES}; found {value!r}."
    raise ValueError(message)


def invalid_geometry_mask(geometries: NDArray[np.object_], /) -> NDArray[np.bool_]:
    """Return which present, nonempty geometries are not valid by ``shapely.is_valid``.

    Parameters
    ----------
    geometries
        One-dimensional array of Shapely geometries. Readers reject missing and
        empty geometries as structural errors before classification.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` for invalid geometries (for example
        self-intersecting rings or bow-ties).

    Raises
    ------
    ValueError
        If the array is not one-dimensional or contains missing or empty
        geometries, which are malformed input rather than invalid geometry.
    """
    if geometries.ndim != 1:
        message = f"geometries must be one-dimensional; found shape {geometries.shape}."
        raise ValueError(message)
    if shapely.is_missing(geometries).any() or shapely.is_empty(geometries).any():
        message = "geometries must not contain missing or empty values; readers reject them as malformed input."
        raise ValueError(message)
    return np.logical_not(shapely.is_valid(geometries))


def raise_invalid_geometries(
    mask: NDArray[np.bool_],
    identifiers: Sequence[object],
    /,
    *,
    context: ReaderErrorContext,
    path: Path | None = None,
) -> None:
    """Raise a contextual error when any geometry is invalid.

    Parameters
    ----------
    mask
        Invalid-geometry mask from :func:`invalid_geometry_mask`.
    identifiers
        Reader-owned identifiers aligned with ``mask``; at most ten invalid ones
        are reported.
    context
        Reader and artifact identity.
    path
        Optional source artifact.

    Raises
    ------
    ReaderFormatError
        If ``mask`` contains any ``True`` value.
    ValueError
        If ``mask`` and ``identifiers`` have different lengths.
    """
    if len(mask) != len(identifiers):
        message = f"mask and identifiers must have equal lengths; found {len(mask)} and {len(identifiers)}."
        raise ValueError(message)
    invalid = np.flatnonzero(mask)
    if invalid.size == 0:
        return
    examples = [identifiers[int(position)] for position in invalid[:_EXAMPLES]]
    raise ReaderFormatError(
        context=context,
        path=path,
        found=f"{invalid.size} invalid geometries (not shapely.is_valid), e.g. {examples!r}",
        expected="valid geometries; invalid_geometries='keep' preserves them and 'drop' removes them",
    )
