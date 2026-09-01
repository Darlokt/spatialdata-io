"""Stereo-seq command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options, parse_json_object

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TypedDict, Unpack

    class _StereoseqOptions(TypedDict):
        dataset_id: str | None
        read_square_bin: bool
        optional_tif: bool
        imread_kwargs: str
        image_models_kwargs: str


@click.command(name="stereoseq")
@dataset_io_options
@click.option("--dataset-id", type=str, default=None, help="Dataset identifier; inferred when omitted.")
@click.option(
    "--read-square-bin", type=bool, default=True, show_default=True, help="Whether to read square-bin expression."
)
@click.option(
    "--optional-tif", type=bool, default=False, show_default=True, help="Whether to read the registered TIFF image."
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
def stereoseq_command(
    input_path: Path,
    output_path: Path,
    **options: Unpack[_StereoseqOptions],
) -> None:
    """Convert Stereo-seq data to SpatialData."""
    # Keep top-level CLI help independent of scientific reader imports.
    from spatialdata_io.readers.stereoseq import stereoseq  # noqa: PLC0415, RUF100

    result = stereoseq(
        input_path,
        dataset_id=options["dataset_id"],
        read_square_bin=options["read_square_bin"],
        optional_tif=options["optional_tif"],
        imread_kwargs=parse_json_object(options["imread_kwargs"], parameter="imread_kwargs"),
        image_models_kwargs=parse_json_object(options["image_models_kwargs"], parameter="image_models_kwargs"),
    )
    result.write(output_path)
