"""Decode explicit Akoya label associations and XY registrations for the CLI.

A single mask uses a path and an optional JSON affine matrix. With several
regions, JSON objects use the same region keys as the Python API. XY matrices
are constructed when a command runs so help does not import SpatialData.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from collections.abc import Callable

    from spatialdata.transformations import Affine

_AFFINE_SIZE = 3


def _json(value: str) -> object:
    def unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result = dict(pairs)
        if len(result) != len(pairs):
            message = "Region names in a JSON mapping must be unique."
            raise click.BadParameter(message)
        return result

    try:
        return json.loads(value, object_pairs_hook=unique_keys)
    except json.JSONDecodeError as error:
        message = "Expected valid JSON."
        raise click.BadParameter(message) from error


def _affine(matrix: object) -> Affine:
    import numpy as np  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.
    from spatialdata.transformations import Affine  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    if (
        not isinstance(matrix, list)
        or len(matrix) != _AFFINE_SIZE
        or any(not isinstance(row, list) or len(row) != _AFFINE_SIZE for row in matrix)
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for row in matrix for item in row)
    ):
        message = "Expected a numeric 3-by-3 XY affine JSON matrix."
        raise click.BadParameter(message)
    try:
        numeric = np.asarray(matrix, dtype=np.float64)
    except OverflowError as error:
        message = "Matrix values must fit finite float64."
        raise click.BadParameter(message) from error
    if not np.isfinite(numeric).all() or not np.array_equal(numeric[2], [0, 0, 1]):
        message = "Expected a finite affine matrix with exact final row [0, 0, 1]."
        raise click.BadParameter(message)
    return Affine(numeric, input_axes=("x", "y"), output_axes=("x", "y"))


def xy_affine(_context: click.Context, _parameter: click.Parameter, value: str | None) -> Affine | None:
    """Decode a single XY affine matrix for the CODEX image-to-table registration."""
    return None if value is None else _affine(_json(value))


def _mask_paths(_context: click.Context, _parameter: click.Parameter, value: str | None) -> str | dict[str, str] | None:
    if value is None or not value.lstrip().startswith("{"):
        return value
    mapping = _json(value)
    if not isinstance(mapping, dict) or not all(isinstance(path, str) and path for path in mapping.values()):
        message = "Expected a JSON mapping from region names to mask paths."
        raise click.BadParameter(message)
    return mapping


def _registrations(
    _context: click.Context, _parameter: click.Parameter, value: str | None
) -> Affine | dict[str, Affine] | None:
    if value is None:
        return None
    parsed = _json(value)
    if isinstance(parsed, dict):
        return {region: _affine(matrix) for region, matrix in parsed.items()}
    return _affine(parsed)


def mask_options[**P](function: Callable[P, None]) -> Callable[P, None]:
    """Add the label path and optional region-specific XY registration options."""
    registered = click.option(
        "--labels-to-image",
        callback=_registrations,
        default=None,
        help="3-by-3 XY affine JSON matrix from mask to image pixels, or JSON object keyed by mask region.",
    )(function)
    return click.option(
        "--labels",
        callback=_mask_paths,
        default=None,
        help="Instance mask TIFF for one selected region, or JSON object mapping regions to mask paths.",
    )(registered)
