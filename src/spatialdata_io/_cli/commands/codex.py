"""CODEX command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options, parse_json_object

if TYPE_CHECKING:
    from pathlib import Path


@click.command(name="codex")
@dataset_io_options
@click.option(
    "--fcs",
    type=bool,
    default=True,
    show_default=True,
    help="Whether an FCS rather than CSV cell table is expected.",
)
@click.option(
    "--imread-kwargs",
    type=str,
    default="{}",
    show_default=True,
    help="JSON object of legacy image-reader options.",
)
def codex_command(input_path: Path, output_path: Path, *, fcs: bool, imread_kwargs: str) -> None:
    """Convert CODEX data to SpatialData."""
    # Keep top-level CLI help independent of scientific reader imports.
    from spatialdata_io.readers.codex import codex  # noqa: PLC0415, RUF100

    result = codex(input_path, fcs=fcs, imread_kwargs=parse_json_object(imread_kwargs, parameter="imread_kwargs"))
    result.write(output_path)
