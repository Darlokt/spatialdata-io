"""DBiT-seq command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options

if TYPE_CHECKING:
    from typing import TypedDict, Unpack

    class _DbitOptions(TypedDict):
        anndata_path: Path | None
        barcode_position: Path | None
        image_path: Path | None
        dataset_id: str | None
        border: bool
        border_scale: float


@click.command(name="dbit")
@dataset_io_options
@click.option(
    "--anndata-path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Path to the counts and metadata file.",
)
@click.option(
    "--barcode-position",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Path to the barcode coordinates file.",
)
@click.option(
    "--image-path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Path to the optional low-resolution image.",
)
@click.option("--dataset-id", type=str, default=None, help="Dataset identifier; inferred when omitted.")
@click.option("--border", type=bool, default=True, show_default=True, help="Whether to add borders to spot geometry.")
@click.option(
    "--border-scale", type=float, default=1, show_default=True, help="Scale applied to spot-geometry borders."
)
def dbit_command(
    input_path: Path,
    output_path: Path,
    **options: Unpack[_DbitOptions],
) -> None:
    """Convert DBiT-seq data to SpatialData."""
    # Keep top-level CLI help independent of scientific reader imports.
    from spatialdata_io.readers.dbit import dbit  # noqa: PLC0415, RUF100

    anndata_path = options["anndata_path"]
    barcode_position = options["barcode_position"]
    image_path = options["image_path"]
    result = dbit(
        input_path,
        anndata_path=None if anndata_path is None else str(anndata_path),
        barcode_position=None if barcode_position is None else str(barcode_position),
        image_path=None if image_path is None else str(image_path),
        dataset_id=options["dataset_id"],
        border=options["border"],
        border_scale=options["border_scale"],
    )
    result.write(output_path)
