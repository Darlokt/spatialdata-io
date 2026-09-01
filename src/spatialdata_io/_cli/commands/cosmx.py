"""CosMx command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options, parse_json_object

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TypedDict, Unpack

    class _CosmxOptions(TypedDict):
        dataset_id: str | None
        transcripts: bool
        imread_kwargs: str
        image_models_kwargs: str


@click.command(name="cosmx")
@dataset_io_options
@click.option("--dataset-id", type=str, default=None, help="Dataset name; inferred when omitted.")
@click.option("--transcripts", type=bool, default=True, show_default=True, help="Whether to load transcripts.")
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
def cosmx_command(
    input_path: Path,
    output_path: Path,
    **options: Unpack[_CosmxOptions],
) -> None:
    """Convert CosMx data to SpatialData."""
    # Keep top-level CLI help independent of scientific reader imports.
    from spatialdata_io.readers.cosmx import cosmx  # noqa: PLC0415, RUF100

    result = cosmx(
        input_path,
        dataset_id=options["dataset_id"],
        transcripts=options["transcripts"],
        imread_kwargs=parse_json_object(options["imread_kwargs"], parameter="imread_kwargs"),
        image_models_kwargs=parse_json_object(options["image_models_kwargs"], parameter="image_models_kwargs"),
    )
    result.write(output_path)
