"""Convert selected seqFISH ROI products to a SpatialData store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options

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
        rois: tuple[str, ...]
        scale_factors: tuple[int, ...]
        chunks: tuple[int, ...]


_IMAGE_RANK = 3


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
@click.option(
    "--scale-factors",
    type=click.IntRange(min=2),
    multiple=True,
    default=(),
    help="Repeat for generated pyramid levels; omitted returns a single scale.",
)
@click.option(
    "--chunks",
    type=click.IntRange(min=1),
    multiple=True,
    default=(),
    help="One size for all axes or three CYX sizes. Labels use YX; omitted keeps native chunks.",
)
def seqfish_command(input_path: Path, output_path: Path, **options: Unpack[_SeqfishOptions]) -> None:
    """Read seqFISH, then write the requested SpatialData store."""
    from spatialdata_io.readers.seqfish import seqfish  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    values = options["chunks"]
    chunks: int | tuple[int, int, int] | None
    if not values:
        chunks = None
    elif len(values) == 1:
        chunks = values[0]
    elif len(values) == _IMAGE_RANK:
        chunks = (values[0], values[1], values[2])
    else:
        message = "Pass --chunks once or three times in CYX order."
        raise click.BadParameter(message, param_hint="--chunks")

    result = seqfish(
        input_path,
        include_images=options["include_images"],
        include_labels=options["include_labels"],
        include_transcripts=options["include_transcripts"],
        include_shapes=options["include_shapes"],
        shape_geometry=options["shape_geometry"],
        rois=options["rois"] or None,
        scale_factors=options["scale_factors"] or None,
        chunks=chunks,
    )
    result.write(output_path)
