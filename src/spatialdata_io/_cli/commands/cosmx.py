"""Convert one CosMx flat-file export to a SpatialData store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import chunks_option, dataset_io_options, scale_factors_option

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TypedDict, Unpack

    class _CosmxOptions(TypedDict):
        """Describe the values normalized by Click."""

        dataset_id: str | None
        include_transcripts: bool
        scale_factors: tuple[int, ...]
        chunks: int | tuple[int, int, int]


@click.command(name="cosmx")
@dataset_io_options
@click.option("--dataset-id", type=str, default=None, help="File prefix; inferred from the count matrix when omitted.")
@click.option(
    "--include-transcripts/--no-include-transcripts", default=True, show_default=True, help="Return lazy transcripts."
)
@scale_factors_option
@chunks_option
def cosmx_command(input_path: Path, output_path: Path, **options: Unpack[_CosmxOptions]) -> None:
    """Read a CosMx export, then write the SpatialData store."""
    from spatialdata_io.readers.cosmx import cosmx  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    result = cosmx(
        input_path,
        dataset_id=options["dataset_id"],
        include_transcripts=options["include_transcripts"],
        scale_factors=options["scale_factors"],
        chunks=options["chunks"],
    )
    result.write(output_path)
