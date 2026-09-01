"""seqFISH command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options, parse_json_object

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TypedDict, Unpack

    class _SeqfishOptions(TypedDict):
        load_images: bool
        load_labels: bool
        load_points: bool
        load_shapes: bool
        cells_as_circles: bool
        rois: tuple[str, ...]
        raster_models_scale_factors: tuple[int, ...]
        imread_kwargs: str


@click.command(name="seqfish")
@dataset_io_options
@click.option("--load-images", type=bool, default=True, show_default=True, help="Whether to load images.")
@click.option("--load-labels", type=bool, default=True, show_default=True, help="Whether to load labels.")
@click.option("--load-points", type=bool, default=True, show_default=True, help="Whether to load transcripts.")
@click.option("--load-shapes", type=bool, default=True, show_default=True, help="Whether to load cell shapes.")
@click.option(
    "--cells-as-circles", type=bool, default=False, show_default=True, help="Whether to represent cells as circles."
)
@click.option("--rois", type=str, multiple=True, default=None, help="Repeated region-of-interest identifiers to load.")
@click.option(
    "--raster-models-scale-factors",
    type=int,
    multiple=True,
    default=None,
    help="Repeated raster-pyramid downsampling factors.",
)
@click.option(
    "--imread-kwargs", type=str, default="{}", show_default=True, help="JSON object of legacy image-reader options."
)
def seqfish_command(
    input_path: Path,
    output_path: Path,
    **options: Unpack[_SeqfishOptions],
) -> None:
    """Convert seqFISH data to SpatialData."""
    # Keep top-level CLI help independent of scientific reader imports.
    from spatialdata_io.readers.seqfish import seqfish  # noqa: PLC0415, RUF100

    rois = options["rois"]
    scale_factors = options["raster_models_scale_factors"]
    result = seqfish(
        input_path,
        load_images=options["load_images"],
        load_labels=options["load_labels"],
        load_points=options["load_points"],
        load_shapes=options["load_shapes"],
        cells_as_circles=options["cells_as_circles"],
        rois=list(rois) if rois else None,
        raster_models_scale_factors=list(scale_factors) if scale_factors else None,
        imread_kwargs=parse_json_object(options["imread_kwargs"], parameter="imread_kwargs"),
    )
    result.write(output_path)
