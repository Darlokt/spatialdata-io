"""Write a processed Akoya scan or directory of regions to a SpatialData Zarr store.

Repeated region and channel-name options preserve command-line order. Omitting
them uses the reader's discovered regions and measured channel names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import chunks_option, path_policy_option, scale_factors_option, scan_io_options

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal, TypedDict, Unpack

    class _PhenoCyclerOptions(TypedDict):
        regions: tuple[str, ...]
        channel_names: tuple[str, ...]
        read_all_bit_depths: bool
        scale_factors: tuple[int, ...]
        chunks: int | tuple[int, int, int]
        path_policy: Literal["strict", "relaxed"]


@click.command(name="phenocycler")
@scan_io_options
@click.option("--regions", multiple=True, default=(), help="Repeat to select scan regions in order.")
@click.option("--channel-names", multiple=True, default=(), help="Repeat once per output channel in source order.")
@click.option(
    "--read-all-bit-depths/--no-read-all-bit-depths",
    default=False,
    show_default=True,
    help="Include both the extended-range and standard images of a paired scan.",
)
@scale_factors_option
@chunks_option
@path_policy_option
def phenocycler_command(input_path: Path, output_path: Path, **options: Unpack[_PhenoCyclerOptions]) -> None:
    """Read PhenoCycler-Fusion, Instrument Manager or quantitative pathology scans."""
    from spatialdata_io.readers.phenocycler import phenocycler  # noqa: PLC0415, RUF100 - Keep CLI help lightweight.

    result = phenocycler(
        input_path,
        regions=options["regions"] or None,
        channel_names=options["channel_names"] or None,
        read_all_bit_depths=options["read_all_bit_depths"],
        chunks=options["chunks"],
        scale_factors=options["scale_factors"],
        path_policy=options["path_policy"],
    )
    result.write(output_path)
