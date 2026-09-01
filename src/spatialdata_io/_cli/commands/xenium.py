"""Xenium command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options, parse_json_object

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TypedDict, Unpack

    class _XeniumOptions(TypedDict):
        cells_boundaries: bool
        nucleus_boundaries: bool
        cells_as_circles: bool
        cells_labels: bool
        nucleus_labels: bool
        transcripts: bool
        morphology_mip: bool
        morphology_focus: bool
        aligned_images: bool
        cells_table: bool
        gex_only: bool
        imread_kwargs: str
        image_models_kwargs: str
        labels_models_kwargs: str


@click.command(name="xenium")
@dataset_io_options
@click.option("--cells-boundaries", type=bool, default=True, show_default=True, help="Whether to read cell boundaries.")
@click.option(
    "--nucleus-boundaries", type=bool, default=True, show_default=True, help="Whether to read nucleus boundaries."
)
@click.option(
    "--cells-as-circles", type=bool, default=False, show_default=True, help="Whether to represent cells as circles."
)
@click.option("--cells-labels", type=bool, default=True, show_default=True, help="Whether to read cell-label rasters.")
@click.option(
    "--nucleus-labels", type=bool, default=True, show_default=True, help="Whether to read nucleus-label rasters."
)
@click.option("--transcripts", type=bool, default=True, show_default=True, help="Whether to read transcripts.")
@click.option(
    "--morphology-mip", type=bool, default=True, show_default=True, help="Whether to read the morphology MIP image."
)
@click.option(
    "--morphology-focus", type=bool, default=True, show_default=True, help="Whether to read morphology focus images."
)
@click.option(
    "--aligned-images", type=bool, default=True, show_default=True, help="Whether to read additional aligned images."
)
@click.option("--cells-table", type=bool, default=True, show_default=True, help="Whether to read the cell table.")
@click.option(
    "--gex-only", type=bool, default=True, show_default=True, help="Whether to retain only gene-expression features."
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
def xenium_command(
    input_path: Path,
    output_path: Path,
    **options: Unpack[_XeniumOptions],
) -> None:
    """Convert Xenium data to SpatialData."""
    # Keep top-level CLI help independent of scientific reader imports.
    from spatialdata_io.readers.xenium import xenium  # noqa: PLC0415, RUF100

    result = xenium(
        input_path,
        cells_boundaries=options["cells_boundaries"],
        nucleus_boundaries=options["nucleus_boundaries"],
        cells_as_circles=options["cells_as_circles"],
        cells_labels=options["cells_labels"],
        nucleus_labels=options["nucleus_labels"],
        transcripts=options["transcripts"],
        morphology_mip=options["morphology_mip"],
        morphology_focus=options["morphology_focus"],
        aligned_images=options["aligned_images"],
        cells_table=options["cells_table"],
        gex_only=options["gex_only"],
        imread_kwargs=parse_json_object(options["imread_kwargs"], parameter="imread_kwargs"),
        image_models_kwargs=parse_json_object(options["image_models_kwargs"], parameter="image_models_kwargs"),
        labels_models_kwargs=parse_json_object(options["labels_models_kwargs"], parameter="labels_models_kwargs"),
    )
    result.write(output_path)
