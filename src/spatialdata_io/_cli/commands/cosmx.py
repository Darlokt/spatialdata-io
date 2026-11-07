"""Convert one CosMx flat-file export to a SpatialData store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import chunks_option, dataset_io_options, path_policy_option, scale_factors_option

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal, TypedDict, Unpack

    class _CosmxOptions(TypedDict):
        """Describe the values normalized by Click."""

        file_prefix: str | None
        dataset_id: str | None
        include_transcripts: bool
        pixel_size_um: float | None
        scale_factors: tuple[int, ...]
        chunks: int | tuple[int, int, int]
        path_policy: Literal["strict", "relaxed"]


@click.command(name="cosmx")
@dataset_io_options
@click.option(
    "--file-prefix",
    type=str,
    default=None,
    help="Prefix of the flat files; inferred from the count matrix when omitted.",
)
@click.option(
    "--dataset-id", type=str, default=None, help="Prefix of the transcripts element name; defaults to the file prefix."
)
@click.option(
    "--include-transcripts/--no-include-transcripts", default=True, show_default=True, help="Return lazy transcripts."
)
@click.option(
    "--pixel-size-um", type=float, default=None, help="Micrometres per global pixel; places elements in 'global'."
)
@scale_factors_option
@chunks_option
@path_policy_option
def cosmx_command(input_path: Path, output_path: Path, **options: Unpack[_CosmxOptions]) -> None:
    """Read a CosMx export, then write the SpatialData store."""
    from spatialdata_io.readers.cosmx import cosmx  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    result = cosmx(
        input_path,
        file_prefix=options["file_prefix"],
        dataset_id=options["dataset_id"],
        include_transcripts=options["include_transcripts"],
        pixel_size_um=options["pixel_size_um"],
        scale_factors=options["scale_factors"],
        chunks=options["chunks"],
        path_policy=options["path_policy"],
    )
    result.write(output_path)
