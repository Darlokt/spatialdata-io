"""Visium command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import chunks_option, dataset_io_options, scale_factors_option

if TYPE_CHECKING:
    from typing import TypedDict, Unpack

    class _VisiumOptions(TypedDict):
        dataset_id: str | None
        counts_source: Path | None
        fullres_image_file: Path | None
        positions_file: Path | None
        scale_factors_file: Path | None
        make_var_names_unique: bool
        data_axes: str | None
        tiff_series: int | None
        scale_factors: tuple[int, ...]
        chunks: int | tuple[int, int, int]


@click.command(name="visium")
@dataset_io_options
@click.option("--dataset-id", type=str, default=None, help="Dataset identifier. Inferred when omitted.")
@click.option(
    "--counts-source",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional HDF5 file or MEX directory. Relative paths resolve below the dataset root.",
)
@click.option(
    "--fullres-image-file",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    default=None,
    help="Optional full-resolution TIFF/BTF, PNG, or JPEG. Relative paths resolve below the dataset root.",
)
@click.option(
    "--positions-file",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    default=None,
    help="Optional tissue-position CSV override. Relative paths resolve below the dataset root.",
)
@click.option(
    "--scale-factors-file",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    default=None,
    help="Optional scale-factor JSON override. Relative paths resolve below the dataset root.",
)
@click.option(
    "--make-var-names-unique/--no-make-var-names-unique",
    default=True,
    show_default=True,
    help="Whether to make feature names unique.",
)
@click.option(
    "--data-axes",
    type=str,
    default=None,
    help="Explicit axes for an ambiguous full-resolution TIFF series, for example CYX or ZCYX.",
)
@click.option(
    "--tiff-series",
    type=click.IntRange(min=0),
    default=None,
    help="Zero-based full-resolution TIFF series. Required when the TIFF contains multiple series.",
)
@scale_factors_option
@chunks_option
def visium_command(
    input_path: Path,
    output_path: Path,
    **options: Unpack[_VisiumOptions],
) -> None:
    """Read one Visium dataset and write it to a SpatialData store."""
    # Keep top-level CLI help independent of scientific reader imports.
    from spatialdata_io.readers.visium import visium  # noqa: PLC0415, RUF100

    result = visium(
        input_path,
        dataset_id=options["dataset_id"],
        counts_source=options["counts_source"],
        fullres_image_file=options["fullres_image_file"],
        positions_file=options["positions_file"],
        scale_factors_file=options["scale_factors_file"],
        make_var_names_unique=options["make_var_names_unique"],
        data_axes=options["data_axes"],
        tiff_series=options["tiff_series"],
        scale_factors=options["scale_factors"],
        chunks=options["chunks"],
    )

    result.write(output_path)
