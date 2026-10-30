"""CODEX command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import chunks_option, dataset_io_options, scale_factors_option

if TYPE_CHECKING:
    from typing import TypedDict, Unpack

    from spatialdata.transformations import Affine

    class _CodexOptions(TypedDict):
        """Describe the values normalized by Click."""

        cell_table_file: Path | None
        image_file: Path | None
        experiment_file: Path | None
        pixel_size_um: float | None
        image_to_table: Affine | None
        data_axes: str | None
        tiff_series: int | None
        scale_factors: tuple[int, ...]
        chunks: int | tuple[int, int, int]


_AFFINE_SIZE = 3


def _image_to_table(_context: click.Context, _parameter: click.Parameter, value: str | None) -> Affine | None:
    """Decode one explicit 3-by-3 XY affine JSON matrix."""
    if value is None:
        return None
    import numpy as np  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.
    from spatialdata.transformations import Affine  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    try:
        matrix: object = json.loads(value)
    except json.JSONDecodeError as error:
        msg = "Expected a 3-by-3 XY affine JSON matrix."
        raise click.BadParameter(msg) from error
    if (
        not isinstance(matrix, list)
        or len(matrix) != _AFFINE_SIZE
        or any(not isinstance(row, list) or len(row) != _AFFINE_SIZE for row in matrix)
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for row in matrix for item in row)
    ):
        message = "Expected a numeric 3-by-3 XY affine JSON matrix."
        raise click.BadParameter(message)
    try:
        numeric = np.asarray(matrix, dtype=np.float64)
    except OverflowError as error:
        message = "Matrix values must fit finite float64."
        raise click.BadParameter(message) from error
    if not np.isfinite(numeric).all() or not np.array_equal(numeric[2], [0, 0, 1]):
        message = "Expected a finite affine matrix with exact final row [0, 0, 1]."
        raise click.BadParameter(message)
    return Affine(numeric, input_axes=("x", "y"), output_axes=("x", "y"))


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
    callback=_image_to_table,
    help="3-by-3 XY affine JSON matrix from raster pixel coordinates to native table XY.",
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
        experiment_file=options["experiment_file"],
        pixel_size_um=options["pixel_size_um"],
        image_to_table=options["image_to_table"],
        data_axes=options["data_axes"],
        tiff_series=options["tiff_series"],
        scale_factors=options["scale_factors"],
        chunks=options["chunks"],
    )
    result.write(output_path)
