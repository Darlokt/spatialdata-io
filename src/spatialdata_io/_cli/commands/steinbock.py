"""Explicit Steinbock source and raster options for the command line."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import chunks_option, dataset_io_options, scale_factors_option

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal, TypedDict, Unpack

    class _SteinbockOptions(TypedDict):
        """Describe Click-converted source choices and image/label options."""

        labels_kind: Literal["deepcell", "ilastik"]
        scale_factors: tuple[int, ...]
        chunks: int | tuple[int, int, int]


@click.command(name="steinbock")
@dataset_io_options
@click.option(
    "--labels-kind",
    type=click.Choice(["deepcell", "ilastik"]),
    default="deepcell",
    show_default=True,
    help="Segmentation-mask family to use.",
)
@scale_factors_option
@chunks_option
def steinbock_command(input_path: Path, output_path: Path, **options: Unpack[_SteinbockOptions]) -> None:
    """Read matched Steinbock samples and write their linked SpatialData store."""
    from spatialdata_io.readers.steinbock import steinbock  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    result = steinbock(
        input_path,
        labels_kind=options["labels_kind"],
        scale_factors=options["scale_factors"],
        chunks=options["chunks"],
    )
    result.write(output_path)
