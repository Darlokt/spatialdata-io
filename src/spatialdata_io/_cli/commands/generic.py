"""Generic raster and GeoJSON command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from typing import TypedDict, Unpack

    class _GenericOptions(TypedDict):
        element_name: str | None
        data_axes: str | None
        tiff_series: int
        tiff_level: int
        chunks: tuple[int, ...]
        coordinate_system: str | None


@click.command(name="generic")
@click.option(
    "--input",
    "input_path",
    "-i",
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to a TIFF/BTF, PNG, JPEG, or GeoJSON file.",
)
@click.option(
    "--output",
    "output_path",
    "-o",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Path to the output SpatialData Zarr store.",
)
@click.option("--name", "element_name", "-n", type=str, help="Name of the element to store.")
@click.option("--data-axes", type=str, help="Image axes: a permutation of yx, cyx, zyx, or czyx.")
@click.option(
    "--tiff-series",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Zero-based TIFF series; only valid for TIFF-family inputs.",
)
@click.option(
    "--tiff-level",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Zero-based TIFF pyramid level; only valid for TIFF-family inputs.",
)
@click.option(
    "--chunks",
    type=click.IntRange(min=1),
    multiple=True,
    default=(),
    help=(
        "Requested image chunks. Pass once to chunk every axis, or repeat in canonical "
        "CYX/CZYX order. If omitted, only Y/X are rechunked to 1024."
    ),
)
@click.option("--coordinate-system", "-c", type=str, help="Target coordinate system.")
def generic_command(
    input_path: Path,
    output_path: Path,
    **options: Unpack[_GenericOptions],
) -> None:
    """Read one generic element and write it to a SpatialData store."""
    # Keep top-level CLI help independent of SpatialData and raster imports.
    from spatialdata_io.converters.generic_to_zarr import generic_to_zarr  # noqa: PLC0415, RUF100

    output_existed = output_path.exists()
    chunks = options["chunks"]
    normalized_chunks: int | tuple[int, ...] | None
    if not chunks:
        normalized_chunks = None
    elif len(chunks) == 1:
        normalized_chunks = chunks[0]
    else:
        normalized_chunks = chunks

    generic_to_zarr(
        path=input_path,
        output=output_path,
        name=options["element_name"],
        data_axes=options["data_axes"],
        coordinate_system=options["coordinate_system"],
        tiff_series=options["tiff_series"],
        tiff_level=options["tiff_level"],
        chunks=normalized_chunks,
    )
    element_name = options["element_name"]
    stored_name = input_path.stem if element_name is None else element_name
    if output_existed:
        click.echo(f"Element {stored_name} written to {output_path}")
    else:
        click.echo(f"Data written to {output_path}")
