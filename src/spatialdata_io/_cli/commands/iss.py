"""ISS command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options

if TYPE_CHECKING:
    from typing import TypedDict, Unpack

    class _IssOptions(TypedDict):
        """Describe the typed values produced by the ISS Click options."""

        raw_image_file: Path
        labels_file: Path
        table_file: Path
        instance_key: str | None
        dataset_id: str
        image_scale_factors: tuple[int, ...]
        labels_scale_factors: tuple[int, ...]
        data_axes: str | None
        chunks: tuple[int, ...]


_IMAGE_RANK = 3


@click.command(name="iss")
@dataset_io_options
@click.option(
    "--raw-image-file",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Raw TIFF-family image; relative paths resolve within the dataset root.",
)
@click.option(
    "--labels-file",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Integer TIFF-family labels; relative paths resolve within the dataset root.",
)
@click.option(
    "--table-file",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Linked h5ad counts and metadata; relative paths resolve within the dataset root.",
)
@click.option(
    "--instance-key", type=str, default=None, help="Observation column containing positive integer label IDs."
)
@click.option("--dataset-id", type=str, default="region", show_default=True, help="Raster element-name prefix.")
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
    "--data-axes", type=str, default=None, help="Source raw-image axes, for example CYX for an unlabeled stack."
)
@click.option(
    "--chunks",
    type=click.IntRange(min=1),
    multiple=True,
    default=(),
    help="One size for all axes or three sizes in CYX order; labels use YX sizes.",
)
def iss_command(input_path: Path, output_path: Path, **options: Unpack[_IssOptions]) -> None:
    """Read ISS image, labels and counts, then write a SpatialData store."""
    from spatialdata_io.readers.iss import iss  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

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
    result = iss(
        input_path,
        raw_image_file=options["raw_image_file"],
        labels_file=options["labels_file"],
        table_file=options["table_file"],
        instance_key=options["instance_key"],
        dataset_id=options["dataset_id"],
        image_scale_factors=options["image_scale_factors"] or None,
        labels_scale_factors=options["labels_scale_factors"] or None,
        data_axes=options["data_axes"],
        chunks=chunks,
    )
    result.write(output_path)
