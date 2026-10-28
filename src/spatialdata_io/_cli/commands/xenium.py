"""Convert selected Xenium products to a SpatialData store."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import click

from spatialdata_io._cli._common import chunks_option, dataset_io_options, scale_factors_option

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TypedDict, Unpack

    class _XeniumOptions(TypedDict):
        """Describe the values normalized by Click."""

        include_table: bool
        feature_types: tuple[str, ...] | Literal["all"] | None
        table_target: Literal["cell_labels", "cell_boundaries", "cell_circles", "none"]
        include_cell_labels: bool
        include_nucleus_labels: bool
        include_cell_boundaries: bool
        cell_geometry: Literal["polygons", "circles", "both"]
        include_nucleus_boundaries: bool
        invalid_geometries: Literal["keep", "drop", "raise"]
        include_transcripts: bool
        include_morphology_focus: bool
        include_morphology_mip: bool
        include_aligned_images: bool
        include_analysis: bool
        scale_factors: tuple[int, ...]
        chunks: int | tuple[int, int, int]


# Feature types written by XOA 1.0-4.x; "Blank Codeword" is the XOA 1.0 name of
# unassigned codewords.
FEATURE_TYPES = (
    "Gene Expression",
    "Protein Expression",
    "Negative Control Probe",
    "Negative Control Codeword",
    "Genomic Control",
    "Unassigned Codeword",
    "Blank Codeword",
    "Deprecated Codeword",
)


def _feature_types(
    _context: click.Context, _parameter: click.Parameter, values: tuple[str, ...]
) -> tuple[str, ...] | Literal["all"] | None:
    """Normalize repeated exact feature types or the sole explicit all-types request."""
    if not values:
        return None
    if values == ("all",):
        return "all"
    if "all" in values:
        message = "Pass exact feature types, or --feature-types all alone."
        raise click.BadParameter(message)
    if len(set(values)) != len(values):
        message = "Feature types must be distinct."
        raise click.BadParameter(message)
    return values


@click.command(name="xenium")
@dataset_io_options
@click.option(
    "--include-table/--no-include-table",
    default=True,
    show_default=True,
    help="Return the cell table.",
)
@click.option(
    "--feature-types",
    type=click.Choice((*FEATURE_TYPES, "all")),
    multiple=True,
    default=(),
    callback=_feature_types,
    help="Repeat exact feature types for X, or pass 'all' alone; omitted selects Gene Expression and any "
    "Protein Expression.",
)
@click.option(
    "--table-target",
    type=click.Choice(["cell_labels", "cell_boundaries", "cell_circles", "none"]),
    default="cell_labels",
    show_default=True,
    help="Element the table annotates; 'none' returns an unlinked table.",
)
@click.option(
    "--include-cell-labels/--no-include-cell-labels",
    default=True,
    show_default=True,
    help="Return the cell segmentation mask.",
)
@click.option(
    "--include-nucleus-labels/--no-include-nucleus-labels",
    default=True,
    show_default=True,
    help="Return the nucleus segmentation mask.",
)
@click.option(
    "--include-cell-boundaries/--no-include-cell-boundaries",
    default=True,
    show_default=True,
    help="Return cells as shapes chosen by --cell-geometry.",
)
@click.option(
    "--cell-geometry",
    type=click.Choice(["polygons", "circles", "both"]),
    default="polygons",
    show_default=True,
    help="Cell shapes: boundary polygons, area-preserving centroid circles, or both.",
)
@click.option(
    "--include-nucleus-boundaries/--no-include-nucleus-boundaries",
    default=True,
    show_default=True,
    help="Return nucleus boundary polygons.",
)
@click.option(
    "--invalid-geometries",
    type=click.Choice(["keep", "drop", "raise"]),
    default="keep",
    show_default=True,
    help="Keep invalid source polygons, drop their entities, or fail.",
)
@click.option(
    "--include-transcripts/--no-include-transcripts",
    default=True,
    show_default=True,
    help="Return lazy transcript points.",
)
@click.option(
    "--include-morphology-focus/--no-include-morphology-focus",
    default=True,
    show_default=True,
    help="Return the 2D morphology focus image.",
)
@click.option(
    "--include-morphology-mip/--no-include-morphology-mip",
    default=False,
    show_default=True,
    help="Return the XOA 1.x maximum-intensity projection.",
)
@click.option(
    "--include-aligned-images/--no-include-aligned-images",
    default=True,
    show_default=True,
    help="Return 10x-named aligned H&E and IF images.",
)
@click.option(
    "--include-analysis/--no-include-analysis",
    default=False,
    show_default=True,
    help="Add 10x clusterings and projections to the table.",
)
@scale_factors_option
@chunks_option
def xenium_command(input_path: Path, output_path: Path, **options: Unpack[_XeniumOptions]) -> None:
    """Read Xenium, then write the requested SpatialData store."""
    from spatialdata_io.readers.xenium import xenium  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    target = options["table_target"]
    result = xenium(
        input_path,
        include_table=options["include_table"],
        feature_types=options["feature_types"],
        table_target=None if target == "none" else target,
        include_cell_labels=options["include_cell_labels"],
        include_nucleus_labels=options["include_nucleus_labels"],
        include_cell_boundaries=options["include_cell_boundaries"],
        cell_geometry=options["cell_geometry"],
        include_nucleus_boundaries=options["include_nucleus_boundaries"],
        invalid_geometries=options["invalid_geometries"],
        include_transcripts=options["include_transcripts"],
        include_morphology_focus=options["include_morphology_focus"],
        include_morphology_mip=options["include_morphology_mip"],
        include_aligned_images=options["include_aligned_images"],
        include_analysis=options["include_analysis"],
        scale_factors=options["scale_factors"],
        chunks=options["chunks"],
    )
    result.write(output_path)
