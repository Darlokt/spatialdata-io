"""Curio command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options

if TYPE_CHECKING:
    from pathlib import Path


@click.command(name="curio")
@dataset_io_options
def curio_command(input_path: Path, output_path: Path) -> None:
    """Convert Curio data to SpatialData."""
    # Keep top-level CLI help independent of scientific reader imports.
    from spatialdata_io.readers.curio import curio  # noqa: PLC0415, RUF100

    curio(input_path).write(output_path)
