"""Convert selected seqFISH ROI products to a SpatialData store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import chunks_option, dataset_io_options, scale_factors_option

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal, TypedDict, Unpack

    class _SeqfishOptions(TypedDict):
        """Describe the values normalized by the Click command."""

        include_images: bool
        include_labels: bool
        include_transcripts: bool
        include_shapes: bool
        shape_geometry: Literal["polygons", "circles", "both"]
        invalid_geometries: Literal["keep", "drop", "raise"]
        rois: tuple[str, ...]
        scale_factors: tuple[int, ...]
        chunks: int | tuple[int, int, int]


@click.command(name="seqfish")
@dataset_io_options
@click.option("--include-images/--no-include-images", default=True, show_default=True, help="Return DAPI images.")
@click.option(
    "--include-labels/--no-include-labels", default=True, show_default=True, help="Return linked segmentations."
)
@click.option(
    "--include-transcripts/--no-include-transcripts", default=True, show_default=True, help="Return transcripts."
)
@click.option("--include-shapes/--no-include-shapes", default=True, show_default=True, help="Return cell shapes.")
@click.option(
    "--shape-geometry",
    type=click.Choice(["polygons", "circles", "both"]),
    default="polygons",
    show_default=True,
    help="Cell geometry representations.",
)
@click.option(
    "--rois", type=str, multiple=True, default=(), help="Repeat ROI prefixes in desired order; omitted selects all."
)
@scale_factors_option
@chunks_option
@click.option(
    "--invalid-geometries",
    type=click.Choice(["keep", "drop", "raise"]),
    default="keep",
    show_default=True,
    help="Keep invalid polygon topology, drop affected cells, or fail.",
)
def seqfish_command(input_path: Path, output_path: Path, **options: Unpack[_SeqfishOptions]) -> None:
    """Read seqFISH, then write the requested SpatialData store."""
    from spatialdata_io.readers.seqfish import seqfish  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    result = seqfish(
        input_path,
        include_images=options["include_images"],
        include_labels=options["include_labels"],
        include_transcripts=options["include_transcripts"],
        include_shapes=options["include_shapes"],
        shape_geometry=options["shape_geometry"],
        invalid_geometries=options["invalid_geometries"],
        rois=options["rois"] or None,
        scale_factors=options["scale_factors"],
        chunks=options["chunks"],
    )
    result.write(output_path)
