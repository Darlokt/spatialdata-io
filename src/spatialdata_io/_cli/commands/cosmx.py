"""Convert one CosMx flat-file export to a SpatialData store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TypedDict, Unpack

    class _CosmxOptions(TypedDict):
        """Describe the values normalized by Click."""

        dataset_id: str | None
        include_transcripts: bool
        scale_factors: tuple[int, ...]
        chunks: tuple[int, ...]


_IMAGE_RANK = 3


@click.command(name="cosmx")
@dataset_io_options
@click.option("--dataset-id", type=str, default=None, help="File prefix; inferred from the count matrix when omitted.")
@click.option(
    "--include-transcripts/--no-include-transcripts", default=True, show_default=True, help="Return lazy transcripts."
)
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
def cosmx_command(input_path: Path, output_path: Path, **options: Unpack[_CosmxOptions]) -> None:
    """Read a CosMx export, then write the SpatialData store."""
    from spatialdata_io.readers.cosmx import cosmx  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

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

    result = cosmx(
        input_path,
        dataset_id=options["dataset_id"],
        include_transcripts=options["include_transcripts"],
        scale_factors=options["scale_factors"] or None,
        chunks=chunks,
    )
    result.write(output_path)
