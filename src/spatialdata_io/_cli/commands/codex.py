"""Write selected CODEX regions to a SpatialData Zarr store.

The command accepts a processed dataset or a CIM scan. An explicit XY affine
maps image pixels into a selected table's native coordinates. Region order
and raster options pass through to the public CODEX reader.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._akoya import mask_options, xy_affine
from spatialdata_io._cli._common import chunks_option, path_policy_option, scale_factors_option, scan_io_options

if TYPE_CHECKING:
    from typing import Literal, TypedDict, Unpack

    from spatialdata.transformations import Affine

    class _CodexOptions(TypedDict):
        """Describe the values normalized by Click."""

        regions: tuple[str, ...]
        read_tables: bool
        compensation: Literal["compensated", "uncompensated"] | None
        cell_table_file: Path | None
        image_file: Path | None
        experiment_file: Path | None
        pixel_size_um: float | None
        image_to_table: Affine | None
        data_axes: str | None
        tiff_series: int | None
        scale_factors: tuple[int, ...]
        chunks: int | tuple[int, int, int]
        labels: str | dict[str, str] | None
        labels_to_image: Affine | dict[str, Affine] | None
        path_policy: Literal["strict", "relaxed"]


@click.command(name="codex")
@scan_io_options
@click.option("--regions", multiple=True, default=(), help="Repeat to select native region IDs in order.")
@click.option("--read-tables/--no-read-tables", default=True, show_default=True, help="Read regional cell tables.")
@click.option(
    "--compensation",
    type=click.Choice(["compensated", "uncompensated"]),
    default=None,
    help="Select one cell table compensation variant across the requested regions.",
)
@click.option(
    "--cell-table-file",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    default=None,
    help="CSV/FCS override. Relative paths resolve within the dataset root.",
)
@click.option(
    "--image-file",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    default=None,
    help="Stitched TIFF-family override. Relative paths resolve within the dataset root.",
)
@click.option(
    "--pixel-size-um", type=float, default=None, help="Associated acquisition/table pixel size in micrometres."
)
@click.option(
    "--experiment-file",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    default=None,
    help="Caller-associated Akoya 1.5/1.7 experiment JSON; relative to the dataset root.",
)
@click.option(
    "--image-to-table",
    type=str,
    default=None,
    callback=xy_affine,
    help="3-by-3 XY affine JSON matrix from raster pixel coordinates to native table XY.",
)
@click.option("--data-axes", type=str, default=None, help="Source TIFF axes, for example CYX for an unlabeled stack.")
@click.option("--tiff-series", type=click.IntRange(min=0), default=None, help="Zero-based TIFF series selection.")
@scale_factors_option
@chunks_option
@mask_options
@path_policy_option
def codex_command(input_path: Path, output_path: Path, **options: Unpack[_CodexOptions]) -> None:
    """Read CODEX regional tables, stitched images or Instrument Manager scans."""
    from spatialdata_io.readers.codex import codex  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    result = codex(
        input_path,
        regions=options["regions"] or None,
        read_tables=options["read_tables"],
        compensation=options["compensation"],
        cell_table_file=options["cell_table_file"],
        image_file=options["image_file"],
        experiment_file=options["experiment_file"],
        pixel_size_um=options["pixel_size_um"],
        image_to_table=options["image_to_table"],
        data_axes=options["data_axes"],
        tiff_series=options["tiff_series"],
        scale_factors=options["scale_factors"],
        chunks=options["chunks"],
        labels=options["labels"],
        labels_to_image=options["labels_to_image"],
        path_policy=options["path_policy"],
    )
    result.write(output_path)
