"""Public orchestration for single-library 10x Genomics Visium output."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata import SpatialData

from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.visium._geometry import make_circles, make_transforms
from spatialdata_io.readers.visium._images import Chunks, make_image_options, read_images
from spatialdata_io.readers.visium._layout import discover_layout
from spatialdata_io.readers.visium._metadata import (
    read_count_metadata,
    read_scale_factors,
    resolve_dataset_id,
)
from spatialdata_io.readers.visium._tables import TableOptions, link_table, prepare_table, read_positions

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def visium(  # noqa: PLR0913, RUF100
    path: str | Path,
    *,
    dataset_id: str | None = None,
    counts_source: str | Path | None = None,
    fullres_image_file: str | Path | None = None,
    positions_file: str | Path | None = None,
    scale_factors_file: str | Path | None = None,
    make_var_names_unique: bool = True,
    data_axes: Sequence[str] | None = None,
    tiff_series: int | None = None,
    chunks: Chunks | None = None,
) -> SpatialData:
    """Read one single-library Space Ranger Visium output directory.

    Supported count sources are 10x HDF5 files and MEX directories containing
    one consistently gzip-compressed or uncompressed
    ``matrix``/``barcodes``/``features`` triplet. Automatic discovery uses the
    standard filtered HDF5 file first, then the standard filtered MEX directory,
    then one legacy dataset-prefixed source. Pass ``counts_source`` explicitly
    to select raw counts or MEX when both representations exist.

    The automatically discovered layout is::

        outs/
        ├── filtered_feature_bc_matrix.h5       # preferred count source
        ├── filtered_feature_bc_matrix/         # alternative MEX source
        │   ├── matrix.mtx[.gz]
        │   ├── barcodes.tsv[.gz]
        │   └── features.tsv[.gz]
        └── spatial/
            ├── tissue_positions.csv            # headered
            │   or tissue_positions_list.csv    # headerless
            ├── scalefactors_json.json
            ├── tissue_hires_image.png          # optional
            └── tissue_lowres_image.png          # optional

    Space Ranger pre-2.0 headerless ``tissue_positions_list.csv``, headered
    legacy files, and 2.0+ ``tissue_positions.csv`` are supported. Position
    rows are aligned to count barcodes without filtering ``in_tissue``. Count
    and feature order are preserved, and expression is returned as a finite
    canonical CSR sparse array. All count feature types are retained in source
    order, including Antibody Capture features when present.

    Full-resolution TIFF/BigTIFF/BTF, PNG, and JPEG inputs are optional.
    Space Ranger hires and lowres PNGs are discovered independently. Pixel
    decoding remains lazy; TIFF construction retains native storage chunks and
    requested chunks add lazy rechunking. Full-resolution images receive up to
    four lazy twofold pyramid reductions. PNG and JPEG computation necessarily
    decodes a complete frame. Tissue imagery is spatially two-dimensional and
    normalized to ``CYX``. Declared channel/sample axes are retained, named
    singleton dimensions are dropped, and ambiguous axes or TIFF series require
    an explicit selection.

    Parameters
    ----------
    path
        Existing Space Ranger ``outs`` directory.
    dataset_id
        Element and coordinate-system identifier. When omitted, use the single
        HDF5 library ID or a legacy count filename prefix.
    counts_source
        Optional HDF5 file or MEX directory. Relative paths resolve below
        ``path``; absolute paths are permitted. Selecting raw counts preserves
        count barcodes whose position rows have ``in_tissue=0``.
    fullres_image_file
        Optional full-resolution tissue image, resolved like ``counts_source``.
    positions_file
        Optional position CSV override, resolved like ``counts_source``.
    scale_factors_file
        Optional scale-factor JSON override, resolved like ``counts_source``.
    make_var_names_unique
        Whether to make duplicate feature names unique once after count import.
    data_axes
        Optional explicit axes for an ambiguous full-resolution TIFF series.
        The names must match the selected array rank and include ``x`` and ``y``.
        This option does not apply to PNG or JPEG inputs.
    tiff_series
        Optional zero-based full-resolution TIFF series. A sole series is
        selected automatically; a multi-series TIFF requires this option.
    chunks
        Requested canonical ``CYX`` model chunks. By default, spatial axes are
        rechunked to 1024 and source channel chunks are retained.

    Returns
    -------
    spatialdata.SpatialData
        Images named ``<dataset_id>_{full,hires,lowres}_image`` when present,
        barcode-indexed spot circles named ``<dataset_id>``, and one linked
        sparse table named ``table``.

    Raises
    ------
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If required paths are absent or have the wrong filesystem kind.
    ValueError
        If layout, schema, numerical metadata, axes, selections, chunks, or
        cross-element membership violate the Visium contract.
    TypeError
        If public option or expression storage types are unsupported.

    Notes
    -----
    The reader never mutates caller-owned scientific containers. Counts,
    position metadata, circles, and the linked table are eager and owned by the
    returned object. Raster metadata inspection is eager and bounded; raster
    pixels remain path-backed and lazy. Coordinates are full-resolution pixels
    with ``x=pxl_col_in_fullres`` and ``y=pxl_row_in_fullres``. The scale-factor
    value ``spot_diameter_fullres`` is divided by two to construct circle radii.
    """
    if not isinstance(make_var_names_unique, bool):
        message = "make_var_names_unique must be a boolean."
        raise TypeError(message)
    image_options = make_image_options(chunks=chunks, data_axes=data_axes, tiff_series=tiff_series)
    layout = discover_layout(
        path,
        counts_source=counts_source,
        fullres_image_file=fullres_image_file,
        positions_file=positions_file,
        scale_factors_file=scale_factors_file,
    )
    count_metadata = read_count_metadata(layout.counts)
    resolved_id = resolve_dataset_id(dataset_id, layout.counts, count_metadata)
    scale_factors = read_scale_factors(layout.scale_factors)
    transforms = make_transforms(scale_factors)
    images = read_images(
        layout,
        dataset_id=resolved_id,
        transforms=transforms,
        options=image_options,
    )
    positions = read_positions(layout)
    prepared = prepare_table(
        layout,
        positions,
        options=TableOptions(
            dataset_id=resolved_id,
            metadata=count_metadata,
            scale_factors=scale_factors,
            make_var_names_unique=make_var_names_unique,
        ),
    )
    circles = make_circles(
        prepared.positions,
        dataset_id=resolved_id,
        radius=scale_factors.spot_diameter / 2.0,
        transforms=transforms,
    )
    table = link_table(prepared, dataset_id=resolved_id, shape_index=circles.index)
    result = SpatialData(images=images, shapes={resolved_id: circles}, tables={"table": table})
    set_reader_provenance(result.attrs, reader="visium", format_version=count_metadata.software_version)
    return result
