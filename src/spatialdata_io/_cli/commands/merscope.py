"""Convert selected MERSCOPE products to a SpatialData store."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import chunks_option, dataset_io_options, scale_factors_option

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
        chunks: int | tuple[int, int, int]


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
@scale_factors_option
@chunks_option
def merscope_command(input_path: Path, output_path: Path, **options: Unpack[_MerscopeOptions]) -> None:
    """Read MERSCOPE, then write the requested SpatialData store."""
    from spatialdata_io.readers.merscope import merscope  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

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
        scale_factors=options["scale_factors"],
        chunks=options["chunks"],
    )
    result.write(output_path)
