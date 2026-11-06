"""ISS command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import chunks_option, dataset_io_options, path_policy_option, scale_factors_option

if TYPE_CHECKING:
    from typing import Literal, TypedDict, Unpack

    class _IssOptions(TypedDict):
        """Describe the typed values produced by the ISS Click options."""

        raw_image_file: Path
        labels_file: Path
        table_file: Path
        instance_key: str | None
        dataset_id: str
        data_axes: str | None
        scale_factors: tuple[int, ...]
        chunks: int | tuple[int, int, int]
        path_policy: Literal["strict", "relaxed"]


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
    "--instance-key",
    type=str,
    default=None,
    help="Observation column holding positive integer label IDs. Default: the observation names.",
)
@click.option("--dataset-id", type=str, default="region", show_default=True, help="Prefix of the raster element names.")
@click.option(
    "--data-axes",
    type=str,
    default=None,
    help="Raw-image axes, for example CYX for a stack without axis metadata or YXC for RGB samples.",
)
@scale_factors_option
@chunks_option
@path_policy_option
def iss_command(input_path: Path, output_path: Path, **options: Unpack[_IssOptions]) -> None:
    """Read an experimental ISS raw image, labels and linked h5ad table, then write a SpatialData store."""
    from spatialdata_io.experimental.iss import iss  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    result = iss(
        input_path,
        raw_image_file=options["raw_image_file"],
        labels_file=options["labels_file"],
        table_file=options["table_file"],
        instance_key=options["instance_key"],
        dataset_id=options["dataset_id"],
        data_axes=options["data_axes"],
        scale_factors=options["scale_factors"],
        chunks=options["chunks"],
        path_policy=options["path_policy"],
    )
    result.write(output_path)
