"""CODEX command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options

_CANONICAL_IMAGE_RANK = 3

if TYPE_CHECKING:
    from typing import TypedDict, Unpack

    class _CodexOptions(TypedDict):
        """Describe Click-converted values before image-chunk normalization."""

        cell_table_file: Path | None
        image_file: Path | None
        data_axes: str | None
        tiff_series: int | None
        chunks: tuple[int, ...]


@click.command(name="codex")
@dataset_io_options
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
@click.option("--data-axes", type=str, default=None, help="Source TIFF axes, for example CYX for an unlabeled stack.")
@click.option("--tiff-series", type=click.IntRange(min=0), default=None, help="Zero-based TIFF series selection.")
@click.option(
    "--chunks",
    type=click.IntRange(min=1),
    multiple=True,
    default=(),
    help="Requested image chunks: one value for all axes or three values in CYX order.",
)
def codex_command(input_path: Path, output_path: Path, **options: Unpack[_CodexOptions]) -> None:
    """Read a CODEX region and write it to a SpatialData store."""
    from spatialdata_io.readers.codex import codex  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    values = options["chunks"]
    chunks: int | tuple[int, int, int] | None
    if not values:
        chunks = None
    elif len(values) == 1:
        chunks = values[0]
    elif len(values) == _CANONICAL_IMAGE_RANK:
        chunks = (values[0], values[1], values[2])
    else:
        message = "--chunks must be passed once for all axes or three times in CYX order."
        raise click.BadParameter(message, param_hint="--chunks")

    result = codex(
        input_path,
        cell_table_file=options["cell_table_file"],
        image_file=options["image_file"],
        data_axes=options["data_axes"],
        tiff_series=options["tiff_series"],
        chunks=chunks,
    )
    result.write(output_path)
