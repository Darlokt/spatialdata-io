"""MCMICRO command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options, parse_json_object

if TYPE_CHECKING:
    from pathlib import Path


@click.command(name="mcmicro")
@dataset_io_options
@click.option(
    "--imread-kwargs", type=str, default="{}", show_default=True, help="JSON object of legacy image-reader options."
)
@click.option(
    "--image-models-kwargs",
    type=str,
    default="{}",
    show_default=True,
    help="JSON object of legacy image-model options.",
)
@click.option(
    "--labels-models-kwargs",
    type=str,
    default="{}",
    show_default=True,
    help="JSON object of legacy labels-model options.",
)
def mcmicro_command(
    input_path: Path,
    output_path: Path,
    *,
    imread_kwargs: str,
    image_models_kwargs: str,
    labels_models_kwargs: str,
) -> None:
    """Convert MCMICRO data to SpatialData."""
    # Keep top-level CLI help independent of scientific reader imports.
    from spatialdata_io.readers.mcmicro import mcmicro  # noqa: PLC0415, RUF100

    result = mcmicro(
        input_path,
        imread_kwargs=parse_json_object(imread_kwargs, parameter="imread_kwargs"),
        image_models_kwargs=parse_json_object(image_models_kwargs, parameter="image_models_kwargs"),
        labels_models_kwargs=parse_json_object(labels_models_kwargs, parameter="labels_models_kwargs"),
    )
    result.write(output_path)
