"""Command converting one Curio Seeker sample export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options, path_policy_option

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal


@click.command(name="curio")
@dataset_io_options
@path_policy_option
def curio_command(input_path: Path, output_path: Path, path_policy: Literal["strict", "relaxed"]) -> None:
    """Read a Curio Seeker sample export and write it to a SpatialData store."""
    # Keep top-level CLI help independent of scientific reader imports.
    from spatialdata_io.readers.curio import curio  # noqa: PLC0415, RUF100

    curio(input_path, path_policy=path_policy).write(output_path)
