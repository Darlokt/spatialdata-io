"""CODEX command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import chunks_option, dataset_io_options, scale_factors_option

if TYPE_CHECKING:
    from typing import TypedDict, Unpack

    class _CodexOptions(TypedDict):
        """Describe the values normalized by Click."""

        cell_table_file: Path | None
        image_file: Path | None
        data_axes: str | None
        tiff_series: int | None
        scale_factors: tuple[int, ...]
        chunks: int | tuple[int, int, int]


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
@scale_factors_option
@chunks_option
def codex_command(input_path: Path, output_path: Path, **options: Unpack[_CodexOptions]) -> None:
    """Read a CODEX region and write it to a SpatialData store."""
    from spatialdata_io.readers.codex import codex  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    result = codex(
        input_path,
        cell_table_file=options["cell_table_file"],
        image_file=options["image_file"],
        data_axes=options["data_axes"],
        tiff_series=options["tiff_series"],
        scale_factors=options["scale_factors"],
        chunks=options["chunks"],
    )
    result.write(output_path)
