"""Convert independently selected Stereo-seq products to SpatialData."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import click

from spatialdata_io._cli._common import chunks_option, dataset_io_options, scale_factors_option

if TYPE_CHECKING:
    from typing import TypedDict, Unpack

    class _StereoseqOptions(TypedDict):
        """Values produced by the command's explicit option parsers."""

        sample_id: str | None
        bin_sizes: tuple[int, ...] | Literal["all"]
        cell_bins: Literal["none", "unadjusted", "adjusted", "both"]
        cell_geometry: Literal["centroids", "polygons"]
        square_counts_file: Path | None
        cell_counts_file: Path | None
        adjusted_cell_counts_file: Path | None
        analysis_files: tuple[Path, ...]
        image_roles: tuple[Literal["registered", "mask", "tissue_cut", "distance_mask", "expression"], ...]
        scale_factors: tuple[int, ...]
        chunks: int | tuple[int, int, int]


def _bin_sizes(
    _context: click.Context, _parameter: click.Parameter, values: tuple[str, ...]
) -> tuple[int, ...] | Literal["all"]:
    """Normalize repeatable positive sizes or the sole explicit all-bin request."""
    if values == ("all",):
        return "all"

    if any(not value.isascii() or not value.isdecimal() or int(value) == 0 for value in values):
        message = "Pass distinct positive integer sizes, or --bin-sizes all alone."
        raise click.BadParameter(message)

    sizes = tuple(int(value) for value in values)
    if len(set(sizes)) != len(sizes):
        message = "Bin sizes must be distinct."
        raise click.BadParameter(message)
    return tuple(sorted(sizes))


@click.command(name="stereoseq")
@dataset_io_options
@click.option("--sample-id", default=None, help="Source sample selector; required for multiple samples.")
@click.option(
    "--bin-sizes",
    multiple=True,
    default=(),
    callback=_bin_sizes,
    help="Repeat stored sizes to read, or pass all alone. Omitted selects no square counts.",
)
@click.option(
    "--cell-bins",
    type=click.Choice(["none", "unadjusted", "adjusted", "both"]),
    default="none",
    show_default=True,
    help="Select independent raw cell populations.",
)
@click.option(
    "--cell-geometry",
    type=click.Choice(["centroids", "polygons"]),
    default="centroids",
    show_default=True,
    help="Stored centers or literal valid contours; polygons are never repaired.",
)
@click.option(
    "--square-counts-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Square GEF source override; requires --bin-sizes.",
)
@click.option(
    "--cell-counts-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Unadjusted cell GEF source override.",
)
@click.option(
    "--adjusted-cell-counts-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Adjusted cell GEF source override.",
)
@click.option(
    "--analysis-files",
    type=click.Path(path_type=Path, dir_okay=False),
    multiple=True,
    default=(),
    help="Repeat exact native H5AD paths, relative to the input root or absolute.",
)
@click.option(
    "--image-roles",
    type=click.Choice(["registered", "mask", "tissue_cut", "distance_mask", "expression"]),
    multiple=True,
    default=(),
    help="Repeat roles; all matching stains are loaded lazily.",
)
@scale_factors_option
@chunks_option
def stereoseq_command(input_path: Path, output_path: Path, **options: Unpack[_StereoseqOptions]) -> None:
    """Select products explicitly, read them, and write the resulting SpatialData."""
    from spatialdata_io.readers.stereoseq import stereoseq  # noqa: PLC0415, RUF100 - Keep command help lightweight.

    result = stereoseq(
        input_path,
        sample_id=options["sample_id"],
        bin_sizes=options["bin_sizes"],
        cell_bins=options["cell_bins"],
        cell_geometry=options["cell_geometry"],
        square_counts_file=options["square_counts_file"],
        cell_counts_file=options["cell_counts_file"],
        adjusted_cell_counts_file=options["adjusted_cell_counts_file"],
        analysis_files=options["analysis_files"],
        image_roles=options["image_roles"],
        scale_factors=options["scale_factors"],
        chunks=options["chunks"],
    )
    result.write(output_path)
