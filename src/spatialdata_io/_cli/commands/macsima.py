"""MACSima command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options, parse_json_object

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TypedDict, Unpack

    class _MacsimaOptions(TypedDict):
        parsing_style: str
        filter_folder_names: tuple[str, ...]
        subset: int | None
        c_subset: int | None
        max_chunk_size: int
        c_chunks_size: int
        multiscale: bool
        transformations: bool
        scale_factors: tuple[int, ...]
        default_scale_factor: int
        nuclei_channel_name: str
        split_threshold_nuclei_channel: int | None
        skip_rounds: tuple[int, ...]
        include_cycle_in_channel_name: bool
        imread_kwargs: str


@click.command(name="macsima")
@dataset_io_options
@click.option(
    "--parsing-style",
    type=click.Choice(["processed_single_folder", "processed_multiple_folders", "raw"]),
    default="processed_single_folder",
    show_default=True,
    help="MACSima directory layout to parse.",
)
@click.option(
    "--filter-folder-names", type=str, multiple=True, default=None, help="Folder names to exclude in multi-folder mode."
)
@click.option("--subset", type=int, default=None, help="Limit image width and height to the first N pixels.")
@click.option("--c-subset", type=int, default=None, help="Limit the image to the first N channels.")
@click.option("--max-chunk-size", type=int, default=1024, show_default=True, help="Maximum spatial chunk size.")
@click.option("--c-chunks-size", type=int, default=1, show_default=True, help="Channel-axis chunk size.")
@click.option("--multiscale", type=bool, default=True, show_default=True, help="Whether to create image pyramids.")
@click.option(
    "--transformations",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to attach pixel-to-micron transforms.",
)
@click.option(
    "--scale-factors", type=int, multiple=True, default=None, help="Repeated image-pyramid downsampling factors."
)
@click.option(
    "--default-scale-factor", type=int, default=2, show_default=True, help="Automatic pyramid downsampling factor."
)
@click.option(
    "--nuclei-channel-name", type=str, default="DAPI", show_default=True, help="Substring identifying nuclei channels."
)
@click.option(
    "--split-threshold-nuclei-channel",
    type=int,
    default=2,
    show_default=True,
    help="Minimum nuclei-channel count that triggers splitting.",
)
@click.option("--skip-rounds", type=int, multiple=True, default=None, help="Repeated round numbers to omit.")
@click.option(
    "--include-cycle-in-channel-name",
    type=bool,
    default=False,
    show_default=True,
    help="Whether channel names include their cycle number.",
)
@click.option(
    "--imread-kwargs", type=str, default="{}", show_default=True, help="JSON object of legacy image-reader options."
)
def macsima_command(
    input_path: Path,
    output_path: Path,
    **options: Unpack[_MacsimaOptions],
) -> None:
    """Convert MACSima data to SpatialData."""
    # Keep top-level CLI help independent of scientific reader imports.
    from spatialdata_io.readers.macsima import macsima  # noqa: PLC0415, RUF100

    filter_folder_names = options["filter_folder_names"]
    scale_factors = options["scale_factors"]
    skip_rounds = options["skip_rounds"]
    result = macsima(
        path=input_path,
        parsing_style=options["parsing_style"],
        filter_folder_names=list(filter_folder_names) if filter_folder_names else None,
        imread_kwargs=parse_json_object(options["imread_kwargs"], parameter="imread_kwargs"),
        subset=options["subset"],
        c_subset=options["c_subset"],
        max_chunk_size=options["max_chunk_size"],
        c_chunks_size=options["c_chunks_size"],
        multiscale=options["multiscale"],
        transformations=options["transformations"],
        scale_factors=list(scale_factors) if scale_factors else None,
        default_scale_factor=options["default_scale_factor"],
        nuclei_channel_name=options["nuclei_channel_name"],
        split_threshold_nuclei_channel=options["split_threshold_nuclei_channel"],
        skip_rounds=list(skip_rounds) if skip_rounds else None,
        include_cycle_in_channel_name=options["include_cycle_in_channel_name"],
    )
    result.write(output_path)
