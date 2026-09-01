"""Shared mechanics for private SpatialData IO commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from collections.abc import Callable


def parse_json_object(value: str, *, parameter: str) -> dict[str, object]:
    """Decode one legacy JSON object option pending its reader cleanup."""
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        message = f"Invalid JSON for {parameter!r}: {error}"
        raise click.BadParameter(message) from error
    if not isinstance(decoded, dict):
        message = f"Invalid JSON for {parameter!r}: expected an object."
        raise click.BadParameter(message)
    return decoded


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
