"""Shared mechanics for private SpatialData IO commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS

if TYPE_CHECKING:
    from collections.abc import Callable

_IMAGE_RANK = 3


def dataset_io_options[**P](function: Callable[P, None]) -> Callable[P, None]:
    """Add the common dataset-root and output-store options."""
    with_output = click.option(
        "--output",
        "output_path",
        "-o",
        type=click.Path(path_type=Path, file_okay=False),
        help="Path to the output SpatialData Zarr store.",
        required=True,
    )(function)
    return click.option(
        "--input",
        "input_path",
        "-i",
        type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
        help="Path to the input dataset directory.",
        required=True,
    )(with_output)


def _chunks(
    _context: click.Context, _parameter: click.Parameter, values: tuple[int, ...]
) -> int | tuple[int, int, int]:
    """Normalize one size for every axis or three ``c, y, x`` sizes."""
    if len(values) == 1:
        return values[0]
    if len(values) == _IMAGE_RANK:
        return (values[0], values[1], values[2])
    message = "Pass --chunks once for every axis or three times in c, y, x order."
    raise click.BadParameter(message)


def scale_factors_option[**P](function: Callable[P, None]) -> Callable[P, None]:
    """Add the shared ``--scale-factors`` option of readers with images or labels."""
    return click.option(
        "--scale-factors",
        type=click.IntRange(min=2),
        multiple=True,
        default=DEFAULT_SCALE_FACTORS,
        help="Repeat once per generated pyramid level; omitted keeps a single scale.",
    )(function)


def chunks_option[**P](function: Callable[P, None]) -> Callable[P, None]:
    """Add the shared ``--chunks`` option of readers with images or labels."""
    return click.option(
        "--chunks",
        type=click.IntRange(min=1),
        multiple=True,
        default=DEFAULT_CHUNKS,
        show_default=True,
        callback=_chunks,
        help="One size for every axis or three c, y, x sizes; labels use the y, x sizes.",
    )(function)
