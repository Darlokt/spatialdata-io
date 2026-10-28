"""Convert one MCMICRO run to a SpatialData store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import chunks_option, dataset_io_options, scale_factors_option

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TypedDict, Unpack

    class _McmicroOptions(TypedDict):
        """Describe the values normalized by Click."""

        scale_factors: tuple[int, ...]
        chunks: int | tuple[int, int, int]


@click.command(name="mcmicro")
@dataset_io_options
@scale_factors_option
@chunks_option
def mcmicro_command(input_path: Path, output_path: Path, **options: Unpack[_McmicroOptions]) -> None:
    """Read MCMICRO outputs, then write the SpatialData store."""
    from spatialdata_io.readers.mcmicro import mcmicro  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    result = mcmicro(input_path, scale_factors=options["scale_factors"], chunks=options["chunks"])
    result.write(output_path)
