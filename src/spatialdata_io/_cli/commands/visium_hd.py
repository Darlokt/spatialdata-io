"""Visium HD command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._cli._common import dataset_io_options, parse_json_object

if TYPE_CHECKING:
    from typing import TypedDict, Unpack

    class _VisiumHdOptions(TypedDict):
        dataset_id: str | None
        filtered_counts_file: bool
        bin_size: tuple[int, ...]
        bins_as_squares: bool
        fullres_image_file: Path | None
        load_all_images: bool
        annotate_table_by_labels: bool
        load_segmentations_only: bool | None
        load_nucleus_segmentations: bool
        var_names_make_unique: bool
        gex_only: bool
        imread_kwargs: str
        image_models_kwargs: str
        anndata_kwargs: str


@click.command(name="visium-hd")
@dataset_io_options
@click.option("--dataset-id", type=str, default=None, help="Dataset identifier; inferred when omitted.")
@click.option(
    "--filtered-counts-file",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to read filtered rather than raw counts.",
)
@click.option("--bin-size", type=int, multiple=True, default=None, help="Repeated spatial bin sizes to load.")
@click.option(
    "--bins-as-squares",
    type=bool,
    default=True,
    show_default=True,
    help="Whether bins are squares rather than circles.",
)
@click.option(
    "--fullres-image-file",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Path to the full-resolution tissue image.",
)
@click.option(
    "--load-all-images",
    type=bool,
    default=False,
    show_default=True,
    help="Whether to load all available image products.",
)
@click.option(
    "--annotate-table-by-labels",
    type=bool,
    default=False,
    show_default=True,
    help="Whether to annotate the table using segmentation labels.",
)
@click.option(
    "--load-segmentations-only",
    type=bool,
    default=None,
    help="Whether to load only segmented-cell products; omission uses the reader's deprecated default.",
)
@click.option(
    "--load-nucleus-segmentations",
    type=bool,
    default=False,
    show_default=True,
    help="Whether to load nucleus segmentations.",
)
@click.option(
    "--var-names-make-unique",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to make feature names unique.",
)
@click.option(
    "--gex-only", type=bool, default=False, show_default=True, help="Whether to retain only gene-expression features."
)
@click.option(
    "--imread-kwargs", type=str, default="{}", show_default=True, help="JSON object of legacy image-reader options."
)
@click.option(
    "--image-models-kwargs",
    type=str,
    default="{}",
    show_default=True,
    help="JSON object of legacy image-model options.",
)
@click.option(
    "--anndata-kwargs", type=str, default="{}", show_default=True, help="JSON object of legacy AnnData-reader options."
)
def visium_hd_command(
    input_path: Path,
    output_path: Path,
    **options: Unpack[_VisiumHdOptions],
) -> None:
    """Convert Visium HD data to SpatialData."""
    # Keep top-level CLI help independent of scientific reader imports.
    from spatialdata_io.readers.visium_hd import visium_hd  # noqa: PLC0415, RUF100

    bin_size = options["bin_size"]
    result = visium_hd(
        path=input_path,
        dataset_id=options["dataset_id"],
        filtered_counts_file=options["filtered_counts_file"],
        load_segmentations_only=options["load_segmentations_only"],
        load_nucleus_segmentations=options["load_nucleus_segmentations"],
        bin_size=list(bin_size) if bin_size else None,
        bins_as_squares=options["bins_as_squares"],
        fullres_image_file=options["fullres_image_file"],
        load_all_images=options["load_all_images"],
        annotate_table_by_labels=options["annotate_table_by_labels"],
        var_names_make_unique=options["var_names_make_unique"],
        gex_only=options["gex_only"],
        imread_kwargs=parse_json_object(options["imread_kwargs"], parameter="imread_kwargs"),
        image_models_kwargs=parse_json_object(options["image_models_kwargs"], parameter="image_models_kwargs"),
        anndata_kwargs=parse_json_object(options["anndata_kwargs"], parameter="anndata_kwargs"),
    )
    result.write(output_path)
