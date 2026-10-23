"""Explicit DBiT source, grid, and raster options for the command line."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options

if TYPE_CHECKING:
    from typing import TypedDict, Unpack

    class _DbitOptions(TypedDict):
        """Describe Click-converted source, grid, and image options."""

        table_file: Path | None
        barcode_positions_file: Path | None
        image_file: Path | None
        dataset_id: str | None
        include_border: bool
        border_scale: float
        data_axes: str | None
        tiff_series: int
        tiff_level: int
        chunks: tuple[int, ...]


_IMAGE_RANK = 3


@click.command(name="dbit")
@dataset_io_options
@click.option(
    "--table-file",
    type=click.Path(path_type=Path, readable=False),
    default=None,
    help="H5AD override. Relative paths resolve below the dataset root.",
)
@click.option(
    "--barcode-positions-file",
    type=click.Path(path_type=Path, readable=False),
    default=None,
    help="Two-column barcode-map override. Relative paths resolve below the dataset root.",
)
@click.option(
    "--image-file",
    type=click.Path(path_type=Path, readable=False),
    default=None,
    help="TIFF/BTF, PNG, or JPEG override. Relative paths resolve below the dataset root.",
)
@click.option("--dataset-id", type=str, default=None, help="Element identifier. Defaults to the H5AD filename stem.")
@click.option(
    "--include-border/--no-include-border",
    default=True,
    show_default=True,
    help="Inset each spot square to leave a border.",
)
@click.option(
    "--border-scale",
    type=click.FloatRange(min=0, max=4, max_open=True),
    default=1.0,
    show_default=True,
    help="Finite border multiplier in [0, 4). Validated even when borders are disabled.",
)
@click.option(
    "--data-axes",
    type=str,
    default=None,
    help="Explicit TIFF source axes when metadata is ambiguous, for example CYX or YXC.",
)
@click.option(
    "--tiff-series",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Zero-based TIFF series to read.",
)
@click.option(
    "--tiff-level",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Zero-based pyramid level within the selected TIFF series.",
)
@click.option(
    "--chunks",
    type=click.IntRange(min=1),
    multiple=True,
    default=(),
    help="One size for all axes or three sizes in CYX order. Defaults to native image chunks.",
)
def dbit_command(input_path: Path, output_path: Path, **options: Unpack[_DbitOptions]) -> None:
    """Read one DBiT dataset and write its linked SpatialData store."""
    # Click's range comparison accepts NaN, so check finiteness explicitly.
    if not math.isfinite(options["border_scale"]):
        message = "--border-scale must be finite and in [0, 4)."
        raise click.BadParameter(message, param_hint="--border-scale")

    values = options["chunks"]
    chunks: int | tuple[int, int, int] | None
    if not values:
        chunks = None
    elif len(values) == 1:
        chunks = values[0]
    elif len(values) == _IMAGE_RANK:
        chunks = (values[0], values[1], values[2])
    else:
        message = "--chunks must be passed once for all axes or three times in CYX order."
        raise click.BadParameter(message, param_hint="--chunks")

    from spatialdata_io.readers.dbit import dbit  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    result = dbit(
        input_path,
        table_file=options["table_file"],
        barcode_positions_file=options["barcode_positions_file"],
        image_file=options["image_file"],
        dataset_id=options["dataset_id"],
        include_border=options["include_border"],
        border_scale=options["border_scale"],
        data_axes=options["data_axes"],
        tiff_series=options["tiff_series"],
        tiff_level=options["tiff_level"],
        chunks=chunks,
    )

    result.write(output_path)
