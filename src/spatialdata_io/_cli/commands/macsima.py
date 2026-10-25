"""Convert processed MACSima exports to a SpatialData store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal, TypedDict, Unpack

    class _MacsimaOptions(TypedDict):
        """Describe the values normalized by Click."""

        parsing_style: Literal["processed_single_folder", "processed_multiple_folders"]
        exclude_folder_names: tuple[str, ...]
        exclude_cycles: tuple[int, ...]
        include_cycle_in_channel_names: bool
        nuclei_channel_name: str
        split_threshold_nuclei_channel: int | None
        include_transformations: bool
        scale_factors: tuple[int, ...]
        chunks: tuple[int, ...]


_IMAGE_RANK = 3


def _split_threshold(_context: click.Context, _parameter: click.Parameter, value: str) -> int | None:
    """Parse a non-negative nuclei-channel count, or ``none`` to never split."""
    text = value.strip()
    if text.lower() == "none":
        return None
    if not text.isdecimal():
        message = f"{value!r} is neither a non-negative integer nor 'none'."
        raise click.BadParameter(message)
    return int(text)


@click.command(name="macsima")
@dataset_io_options
@click.option(
    "--parsing-style",
    type=click.Choice(["processed_single_folder", "processed_multiple_folders"]),
    default="processed_single_folder",
    show_default=True,
    help="One stack for the folder, or one stack per immediate subfolder.",
)
@click.option(
    "--exclude-folder-names", type=str, multiple=True, default=(), help="Repeat exact subfolder names to omit."
)
@click.option(
    "--exclude-cycles", type=click.IntRange(min=0), multiple=True, default=(), help="Repeat acquisition cycles to omit."
)
@click.option(
    "--include-cycle-in-channel-names/--no-include-cycle-in-channel-names",
    default=False,
    show_default=True,
    help="Prefix channel names with their cycle.",
)
@click.option(
    "--nuclei-channel-name", type=str, default="DAPI", show_default=True, help="Exact marker name of the nuclear stain."
)
@click.option(
    "--split-threshold-nuclei-channel",
    type=str,
    default="2",
    metavar="COUNT|none",
    callback=_split_threshold,
    show_default=True,
    help="Split nuclei channels into their own stack above this count; 'none' never splits.",
)
@click.option(
    "--include-transformations/--no-include-transformations",
    default=True,
    show_default=True,
    help="Scale pixels to micrometers.",
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
    help="One size for every axis or three CYX sizes; omitted uses 1, 1024, 1024.",
)
def macsima_command(input_path: Path, output_path: Path, **options: Unpack[_MacsimaOptions]) -> None:
    """Read processed MACSima exports, then write the SpatialData store."""
    from spatialdata_io.readers.macsima import macsima  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    sizes = options["chunks"]
    chunks: int | tuple[int, int, int]
    if not sizes:
        chunks = (1, 1024, 1024)
    elif len(sizes) == 1:
        chunks = sizes[0]
    elif len(sizes) == _IMAGE_RANK:
        chunks = (sizes[0], sizes[1], sizes[2])
    else:
        message = "Pass --chunks once or three times in CYX order."
        raise click.BadParameter(message, param_hint="--chunks")

    result = macsima(
        input_path,
        parsing_style=options["parsing_style"],
        exclude_folder_names=options["exclude_folder_names"],
        exclude_cycles=options["exclude_cycles"],
        include_cycle_in_channel_names=options["include_cycle_in_channel_names"],
        nuclei_channel_name=options["nuclei_channel_name"],
        split_threshold_nuclei_channel=options["split_threshold_nuclei_channel"],
        include_transformations=options["include_transformations"],
        scale_factors=options["scale_factors"] or None,
        chunks=chunks,
    )
    result.write(output_path)
