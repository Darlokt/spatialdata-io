"""ISS command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options, parse_json_object

if TYPE_CHECKING:
    from typing import TypedDict, Unpack

    class _IssOptions(TypedDict):
        raw_relative_path: Path
        labels_relative_path: Path
        h5ad_relative_path: Path
        instance_key: str | None
        dataset_id: str
        multiscale_image: bool
        multiscale_labels: bool
        imread_kwargs: str
        image_models_kwargs: str
        labels_models_kwargs: str


@click.command(name="iss")
@dataset_io_options
@click.option(
    "--raw-relative-path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to the raw raster image, relative to the dataset root or absolute.",
)
@click.option(
    "--labels-relative-path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to the labels image, relative to the dataset root or absolute.",
)
@click.option(
    "--h5ad-relative-path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to the counts and metadata file, relative to the dataset root or absolute.",
)
@click.option("--instance-key", type=str, default=None, help="Observation column containing cell identifiers.")
@click.option("--dataset-id", type=str, default="region", show_default=True, help="Dataset identifier.")
@click.option(
    "--multiscale-image", type=bool, default=True, show_default=True, help="Whether to create an image pyramid."
)
@click.option(
    "--multiscale-labels", type=bool, default=True, show_default=True, help="Whether to create a labels pyramid."
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
@click.option(
    "--labels-models-kwargs",
    type=str,
    default="{}",
    show_default=True,
    help="JSON object of legacy labels-model options.",
)
def iss_command(
    input_path: Path,
    output_path: Path,
    **options: Unpack[_IssOptions],
) -> None:
    """Convert experimental ISS data to SpatialData."""
    # Keep top-level CLI help independent of scientific reader imports.
    from spatialdata_io.readers.iss import iss  # noqa: PLC0415, RUF100

    result = iss(
        input_path,
        options["raw_relative_path"],
        options["labels_relative_path"],
        options["h5ad_relative_path"],
        instance_key=options["instance_key"],
        dataset_id=options["dataset_id"],
        multiscale_image=options["multiscale_image"],
        multiscale_labels=options["multiscale_labels"],
        imread_kwargs=parse_json_object(options["imread_kwargs"], parameter="imread_kwargs"),
        image_models_kwargs=parse_json_object(options["image_models_kwargs"], parameter="image_models_kwargs"),
        labels_models_kwargs=parse_json_object(options["labels_models_kwargs"], parameter="labels_models_kwargs"),
    )
    result.write(output_path)
