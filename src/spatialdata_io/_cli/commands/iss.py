"""ISS command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import chunks_option, dataset_io_options, scale_factors_option

if TYPE_CHECKING:
    from typing import TypedDict, Unpack

    class _IssOptions(TypedDict):
        """Describe the typed values produced by the ISS Click options."""

        raw_image_file: Path
        labels_file: Path
        table_file: Path
        instance_key: str | None
        dataset_id: str
        scale_factors: tuple[int, ...]
        data_axes: str | None
        chunks: int | tuple[int, int, int]


@click.command(name="iss")
@dataset_io_options
@click.option(
    "--raw-image-file",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Raw TIFF-family image. Relative paths resolve within the dataset root.",
)
@click.option(
    "--labels-file",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Integer TIFF-family labels. Relative paths resolve within the dataset root.",
)
@click.option(
    "--table-file",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Linked h5ad counts and metadata. Relative paths resolve within the dataset root.",
)
@click.option(
    "--instance-key", type=str, default=None, help="Observation column containing positive integer label IDs."
)
@click.option("--dataset-id", type=str, default="region", show_default=True, help="Raster element-name prefix.")
@click.option(
    "--data-axes", type=str, default=None, help="Source raw-image axes, for example CYX for an unlabeled stack."
)
@scale_factors_option
@chunks_option
def iss_command(input_path: Path, output_path: Path, **options: Unpack[_IssOptions]) -> None:
    """Read ISS image, labels and counts, then write a SpatialData store."""
    from spatialdata_io.readers.iss import iss  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    result = iss(
        input_path,
        raw_image_file=options["raw_image_file"],
        labels_file=options["labels_file"],
        table_file=options["table_file"],
        instance_key=options["instance_key"],
        dataset_id=options["dataset_id"],
        scale_factors=options["scale_factors"],
        data_axes=options["data_axes"],
        chunks=options["chunks"],
    )
    result.write(output_path)
