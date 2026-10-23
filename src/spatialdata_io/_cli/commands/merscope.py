"""Convert selected MERSCOPE products to a SpatialData store."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options

if TYPE_CHECKING:
    from typing import TypedDict, Unpack

    class _MerscopeOptions(TypedDict):
        """Describe the values normalized by Click."""

        vpt_directory: Path | None
        z_layers: tuple[int, ...]
        region_name: str | None
        slide_name: str | None
        include_transcripts: bool
        include_cell_boundaries: bool
        include_table: bool
        include_images: bool
        scale_factors: tuple[int, ...]
        chunks: tuple[int, ...]


_IMAGE_RANK = 3


@click.command(name="merscope")
@dataset_io_options
@click.option(
    "--vpt-directory",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="VPT analysis directory; relative paths resolve against the input dataset.",
)
@click.option(
    "--z-layers",
    type=click.IntRange(min=0),
    multiple=True,
    default=(),
    help="Repeat z indices in desired order; omitted selects all discovered layers.",
)
@click.option("--region-name", type=str, default=None, help="Region name; inferred from the input directory.")
@click.option("--slide-name", type=str, default=None, help="Slide name; inferred from the input parent directory.")
@click.option(
    "--include-transcripts/--no-include-transcripts", default=True, show_default=True, help="Return lazy transcripts."
)
@click.option(
    "--include-cell-boundaries/--no-include-cell-boundaries",
    default=True,
    show_default=True,
    help="Return the union of every z-slice boundary for each cell.",
)
@click.option(
    "--include-table/--no-include-table", default=True, show_default=True, help="Return sparse counts and controls."
)
@click.option("--include-images/--no-include-images", default=True, show_default=True, help="Return mosaic images.")
@click.option(
    "--scale-factors",
    type=click.IntRange(min=2),
    multiple=True,
    default=(),
    help="Repeat factors for generated pyramid levels; omitted keeps a single scale.",
)
@click.option(
    "--chunks",
    type=click.IntRange(min=1),
    multiple=True,
    default=(),
    help="One size for every axis or three CYX sizes; omitted keeps native chunks.",
)
def merscope_command(input_path: Path, output_path: Path, **options: Unpack[_MerscopeOptions]) -> None:
    """Read MERSCOPE, then write the requested SpatialData store."""
    from spatialdata_io.readers.merscope import merscope  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    sizes = options["chunks"]
    chunks: int | tuple[int, int, int] | None
    if not sizes:
        chunks = None
    elif len(sizes) == 1:
        chunks = sizes[0]
    elif len(sizes) == _IMAGE_RANK:
        chunks = (sizes[0], sizes[1], sizes[2])
    else:
        message = "Pass --chunks once or three times in CYX order."
        raise click.BadParameter(message, param_hint="--chunks")

    result = merscope(
        input_path,
        vpt_directory=options["vpt_directory"],
        z_layers=options["z_layers"] or None,
        region_name=options["region_name"],
        slide_name=options["slide_name"],
        include_transcripts=options["include_transcripts"],
        include_cell_boundaries=options["include_cell_boundaries"],
        include_table=options["include_table"],
        include_images=options["include_images"],
        scale_factors=options["scale_factors"] or None,
        chunks=chunks,
    )
    result.write(output_path)
