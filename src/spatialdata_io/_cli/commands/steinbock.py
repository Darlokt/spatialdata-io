"""Explicit Steinbock source and raster options for the command line."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal, TypedDict, Unpack

    class _SteinbockOptions(TypedDict):
        """Describe Click-converted source choices and image/label options."""

        labels_kind: Literal["deepcell", "ilastik"]
        image_scale_factors: tuple[int, ...]
        labels_scale_factors: tuple[int, ...]
        chunks: tuple[int, ...]


_IMAGE_RANK = 3


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
    "--image-scale-factors",
    type=click.IntRange(min=2),
    multiple=True,
    default=(),
    help="Repeat for each generated image-pyramid level. No pyramid by default.",
)
@click.option(
    "--labels-scale-factors",
    type=click.IntRange(min=2),
    multiple=True,
    default=(),
    help="Repeat for each label-pyramid level; requires <=32-bit integer masks.",
)
@click.option(
    "--chunks",
    type=click.IntRange(min=1),
    multiple=True,
    default=(),
    help="One size for all axes or three sizes in CYX order; labels use YX sizes.",
)
def steinbock_command(input_path: Path, output_path: Path, **options: Unpack[_SteinbockOptions]) -> None:
    """Read matched Steinbock samples and write their linked SpatialData store."""
    from spatialdata_io.readers.steinbock import steinbock  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    values = options["chunks"]
    chunks: int | tuple[int, int, int] | None
    if not values:
        chunks = None
    elif len(values) == 1:
        chunks = values[0]
    elif len(values) == _IMAGE_RANK:
        chunks = (values[0], values[1], values[2])
    else:
        message = "--chunks must be passed once for all axes or three times in CYX order."
        raise click.BadParameter(message, param_hint="--chunks")

    result = steinbock(
        input_path,
        labels_kind=options["labels_kind"],
        image_scale_factors=options["image_scale_factors"] or None,
        labels_scale_factors=options["labels_scale_factors"] or None,
        chunks=chunks,
    )
    result.write(output_path)
