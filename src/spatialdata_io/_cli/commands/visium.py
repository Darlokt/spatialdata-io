"""Visium command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options, parse_json_object

if TYPE_CHECKING:
    from typing import TypedDict, Unpack

    class _VisiumOptions(TypedDict):
        dataset_id: str | None
        counts_file: str
        fullres_image_file: Path | None
        tissue_positions_file: Path | None
        scalefactors_file: Path | None
        var_names_make_unique: bool
        imread_kwargs: str
        image_models_kwargs: str


@click.command(name="visium")
@dataset_io_options
@click.option("--dataset-id", type=str, default=None, help="Dataset identifier; inferred when omitted.")
@click.option(
    "--counts-file",
    type=str,
    default="filtered_feature_bc_matrix.h5",
    show_default=True,
    help="Counts filename inside the dataset root.",
)
@click.option(
    "--fullres-image-file",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Path to the full-resolution tissue image.",
)
@click.option(
    "--tissue-positions-file",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Path to the tissue-position table.",
)
@click.option(
    "--scalefactors-file",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Path to the scale-factor JSON file.",
)
@click.option(
    "--var-names-make-unique",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to make feature names unique.",
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
def visium_command(
    input_path: Path,
    output_path: Path,
    **options: Unpack[_VisiumOptions],
) -> None:
    """Convert Visium data to SpatialData."""
    # Keep top-level CLI help independent of scientific reader imports.
    from spatialdata_io.readers.visium import visium  # noqa: PLC0415, RUF100

    result = visium(
        input_path,
        dataset_id=options["dataset_id"],
        counts_file=options["counts_file"],
        fullres_image_file=options["fullres_image_file"],
        tissue_positions_file=options["tissue_positions_file"],
        scalefactors_file=options["scalefactors_file"],
        var_names_make_unique=options["var_names_make_unique"],
        imread_kwargs=parse_json_object(options["imread_kwargs"], parameter="imread_kwargs"),
        image_models_kwargs=parse_json_object(options["image_models_kwargs"], parameter="image_models_kwargs"),
    )
    result.write(output_path)
