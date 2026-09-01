"""Steinbock command."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import click

from spatialdata_io._cli._common import dataset_io_options, parse_json_object

if TYPE_CHECKING:
    from pathlib import Path


@click.command(name="steinbock")
@dataset_io_options
@click.option(
    "--labels-kind",
    type=click.Choice(["deepcell", "ilastik"]),
    default="deepcell",
    show_default=True,
    help="Segmentation-mask family to use.",
)
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
def steinbock_command(
    input_path: Path,
    output_path: Path,
    *,
    labels_kind: Literal["deepcell", "ilastik"],
    imread_kwargs: str,
    image_models_kwargs: str,
) -> None:
    """Convert Steinbock data to SpatialData."""
    # Keep top-level CLI help independent of scientific reader imports.
    from spatialdata_io.readers.steinbock import steinbock  # noqa: PLC0415, RUF100

    result = steinbock(
        input_path,
        labels_kind=labels_kind,
        imread_kwargs=parse_json_object(imread_kwargs, parameter="imread_kwargs"),
        image_models_kwargs=parse_json_object(image_models_kwargs, parameter="image_models_kwargs"),
    )
    result.write(output_path)
