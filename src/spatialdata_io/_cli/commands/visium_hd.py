"""Convert selected Visium HD products with explicit typed options."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import click

from spatialdata_io._cli._common import chunks_option, dataset_io_options, scale_factors_option

if TYPE_CHECKING:
    from typing import TypedDict, Unpack

    class _VisiumHDOptions(TypedDict):
        """Values validated by Click and the reader's option boundary."""

        bin_sizes: tuple[int, ...] | Literal["all"]
        bin_geometry: Literal["squares", "labels"]
        bin_counts_kind: Literal["filtered", "raw"]
        include_cell_boundaries: bool | None
        include_nucleus_boundaries: bool | None
        include_nucleus_table: bool
        cell_counts_kind: Literal["filtered", "raw"]
        invalid_geometries: Literal["keep", "drop", "raise"]
        include_images: bool
        include_cytassist_image: bool
        fullres_image_file: Path | None
        data_axes: tuple[str, ...] | None
        tiff_series: int | None
        make_var_names_unique: bool
        scale_factors: tuple[int, ...]
        chunks: int | tuple[int, int, int]


def _bin_sizes(
    _context: click.Context, _parameter: click.Parameter, values: tuple[str, ...]
) -> tuple[int, ...] | Literal["all"]:
    """Parse repeated sizes, all, or none for a segmentation-only workflow."""
    if values == ("all",):
        return "all"
    if values == ("none",):
        return ()

    if any(not value.isascii() or not value.isdecimal() or int(value) == 0 or int(value) % 2 for value in values):
        message = "Pass positive even sizes, or all or none alone."
        raise click.BadParameter(message)

    sizes = tuple(int(value) for value in values)
    if len(set(sizes)) != len(sizes):
        message = "Bin sizes must be distinct."
        raise click.BadParameter(message)

    return tuple(sorted(sizes))


def _axes(_context: click.Context, _parameter: click.Parameter, values: tuple[str, ...]) -> tuple[str, ...] | None:
    """Treat an omitted axis override as metadata-driven selection."""
    return values or None


@click.command(name="visium-hd")
@dataset_io_options
@click.option(
    "--bin-sizes",
    multiple=True,
    default=("all",),
    show_default=True,
    callback=_bin_sizes,
    help="Repeat sizes, or all/none alone.",
)
@click.option(
    "--bin-geometry",
    type=click.Choice(["squares", "labels"]),
    default="squares",
    show_default=True,
    help="Represent nominal bins as exact squares or a grid of labels.",
)
@click.option(
    "--bin-counts-kind",
    type=click.Choice(["filtered", "raw"]),
    default="filtered",
    show_default=True,
    help="Select the bin matrix population.",
)
@click.option(
    "--include-cell-boundaries/--no-include-cell-boundaries", default=None, help="Default: read when present."
)
@click.option(
    "--include-nucleus-boundaries/--no-include-nucleus-boundaries", default=None, help="Default: read when present."
)
@click.option(
    "--include-nucleus-table/--no-include-nucleus-table",
    default=False,
    show_default=True,
    help="Aggregate raw 2 micrometre counts into nuclei.",
)
@click.option(
    "--cell-counts-kind",
    type=click.Choice(["filtered", "raw"]),
    default="filtered",
    show_default=True,
    help="Select the cell matrix population.",
)
@click.option(
    "--invalid-geometries",
    type=click.Choice(["keep", "drop", "raise"]),
    default="keep",
    show_default=True,
    help="Policy for invalid source polygons.",
)
@click.option(
    "--include-images/--no-include-images", default=True, show_default=True, help="Include available microscope images."
)
@click.option(
    "--include-cytassist-image/--no-include-cytassist-image",
    default=False,
    show_default=True,
    help="Include the lazily warped CytAssist image.",
)
@click.option(
    "--fullres-image-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Full-resolution image override relative to input or absolute.",
)
@click.option("--data-axes", multiple=True, default=(), callback=_axes, help="Repeat axis names in source array order.")
@click.option("--tiff-series", type=click.IntRange(min=0), default=None, help="Full-resolution TIFF series index.")
@click.option(
    "--make-var-names-unique/--no-make-var-names-unique",
    default=True,
    show_default=True,
    help="Make duplicate feature symbols unique in all tables.",
)
@scale_factors_option
@chunks_option
def visium_hd_command(input_path: Path, output_path: Path, **options: Unpack[_VisiumHDOptions]) -> None:
    """Read Space Ranger HD outputs and write a SpatialData Zarr store."""
    from spatialdata_io.readers.visium_hd import visium_hd  # noqa: PLC0415, RUF100 - Keep command help lightweight.

    result = visium_hd(
        input_path,
        bin_sizes=options["bin_sizes"],
        bin_geometry=options["bin_geometry"],
        bin_counts_kind=options["bin_counts_kind"],
        include_cell_boundaries=options["include_cell_boundaries"],
        include_nucleus_boundaries=options["include_nucleus_boundaries"],
        include_nucleus_table=options["include_nucleus_table"],
        cell_counts_kind=options["cell_counts_kind"],
        invalid_geometries=options["invalid_geometries"],
        include_images=options["include_images"],
        include_cytassist_image=options["include_cytassist_image"],
        fullres_image_file=options["fullres_image_file"],
        data_axes=options["data_axes"],
        tiff_series=options["tiff_series"],
        make_var_names_unique=options["make_var_names_unique"],
        scale_factors=options["scale_factors"],
        chunks=options["chunks"],
    )
    result.write(output_path)
