"""Convert one MCMICRO run to a SpatialData store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TypedDict, Unpack

    class _McmicroOptions(TypedDict):
        """Describe the values normalized by Click."""

        scale_factors: tuple[int, ...]
        chunks: tuple[int, ...]


_IMAGE_RANK = 3


@click.command(name="mcmicro")
@dataset_io_options
@click.option(
    "--scale-factors",
    type=click.IntRange(min=2),
    multiple=True,
    default=(),
    help="Repeat factors for generated pyramid levels; omitted keeps a single scale.",
)
@click.option(
    "--chunks",
    type=click.IntRange(min=1),
    multiple=True,
    default=(),
    help="One size for every axis or three CYX sizes; omitted keeps native chunks.",
)
def mcmicro_command(input_path: Path, output_path: Path, **options: Unpack[_McmicroOptions]) -> None:
    """Read MCMICRO outputs, then write the SpatialData store."""
    from spatialdata_io.readers.mcmicro import mcmicro  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    sizes = options["chunks"]
    chunks: int | tuple[int, int, int] | None
    if not sizes:
        chunks = None
    elif len(sizes) == 1:
        chunks = sizes[0]
    elif len(sizes) == _IMAGE_RANK:
        chunks = (sizes[0], sizes[1], sizes[2])
    else:
        message = "Pass --chunks once or three times in CYX order."
        raise click.BadParameter(message, param_hint="--chunks")

    result = mcmicro(input_path, scale_factors=options["scale_factors"] or None, chunks=chunks)
    result.write(output_path)
