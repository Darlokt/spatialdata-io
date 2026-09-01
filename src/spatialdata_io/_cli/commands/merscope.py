"""MERSCOPE command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import click

from spatialdata_io._cli._common import dataset_io_options, parse_json_object

if TYPE_CHECKING:
    from typing import TypedDict, Unpack

    class _MerscopeOptions(TypedDict):
        vpt_outputs: Path | None
        z_layers: int
        region_name: str | None
        slide_name: str | None
        backend: Literal["dask_image", "rioxarray"] | None
        transcripts: bool
        cells_boundaries: bool
        cells_table: bool
        mosaic_images: bool
        imread_kwargs: str
        image_models_kwargs: str


@click.command(name="merscope")
@dataset_io_options
@click.option(
    "--vpt-outputs",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Directory containing optional Vizgen post-processing outputs.",
)
@click.option("--z-layers", type=int, default=3, show_default=True, help="Z-layer index to read.")
@click.option("--region-name", type=str, default=None, help="Region-of-interest name; inferred when omitted.")
@click.option("--slide-name", type=str, default=None, help="Slide or run name; inferred when omitted.")
@click.option(
    "--backend",
    type=click.Choice(["dask_image", "rioxarray"]),
    default=None,
    help="Legacy raster backend selection.",
)
@click.option("--transcripts", type=bool, default=True, show_default=True, help="Whether to read transcripts.")
@click.option("--cells-boundaries", type=bool, default=True, show_default=True, help="Whether to read cell boundaries.")
@click.option("--cells-table", type=bool, default=True, show_default=True, help="Whether to read the cell table.")
@click.option("--mosaic-images", type=bool, default=True, show_default=True, help="Whether to read mosaic images.")
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
def merscope_command(
    input_path: Path,
    output_path: Path,
    **options: Unpack[_MerscopeOptions],
) -> None:
    """Convert MERSCOPE data to SpatialData."""
    # Keep top-level CLI help independent of scientific reader imports.
    from spatialdata_io.readers.merscope import merscope  # noqa: PLC0415, RUF100

    result = merscope(
        input_path,
        vpt_outputs=options["vpt_outputs"],
        z_layers=options["z_layers"],
        region_name=options["region_name"],
        slide_name=options["slide_name"],
        backend=options["backend"],
        transcripts=options["transcripts"],
        cells_boundaries=options["cells_boundaries"],
        cells_table=options["cells_table"],
        mosaic_images=options["mosaic_images"],
        imread_kwargs=parse_json_object(options["imread_kwargs"], parameter="imread_kwargs"),
        image_models_kwargs=parse_json_object(options["image_models_kwargs"], parameter="image_models_kwargs"),
    )
    result.write(output_path)
