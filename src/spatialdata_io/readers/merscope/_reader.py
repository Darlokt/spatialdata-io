"""Assemble calibrated Vizgen elements from one instrument or VPT layout."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from spatialdata import SpatialData

from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.merscope._geometry import read_boundaries
from spatialdata_io.readers.merscope._images import make_raster_options, read_images
from spatialdata_io.readers.merscope._layout import discover_layout, make_selection
from spatialdata_io.readers.merscope._tables import finalize_table, read_feature_names, read_table
from spatialdata_io.readers.merscope._transcripts import read_transcripts
from spatialdata_io.readers.merscope._transforms import read_transform

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from anndata import AnnData
    from dask.dataframe import DataFrame
    from geopandas import GeoDataFrame


def merscope(
    path: str | Path,
    *,
    vpt_directory: str | Path | None = None,
    z_layers: Sequence[int] | None = None,
    region_name: str | None = None,
    slide_name: str | None = None,
    include_transcripts: bool = True,
    include_cell_boundaries: bool = True,
    include_table: bool = True,
    include_images: bool = True,
    scale_factors: Sequence[int] | None = None,
    chunks: int | tuple[int, int, int] | None = None,
) -> SpatialData:
    """Read the MERSCOPE instrument and Cellpose/Watershed VPT export layouts.

    Instrument artifacts are ``cell_by_gene.csv``, ``cell_metadata.csv``,
    ``cell_boundaries.parquet`` and ``detected_transcripts.csv``. Mosaic TIFFs
    and ``micron_to_mosaic_pixel_transform.csv`` live in ``images/``.
    VPT replaces cell artifacts with its count/metadata CSVs and exactly one
    ``cellpose_micron_space.parquet`` or ``watershed_micron_space.parquet``.
    Only requested products and their dependencies are required. Filenames do
    not establish a software release, so no release version is inferred.

    Parameters
    ----------
    path
        Instrument region directory.
    vpt_directory
        Optional VPT analysis directory. Relative overrides resolve against
        the region root; absolute overrides are permitted.
    z_layers
        Unique nonnegative image-stack indices in desired order. None selects
        all discovered layers in ascending order. Every selected layer must
        contain the same lexically ordered stains. A scalar is not accepted.
    region_name, slide_name
        Element naming components, defaulting to the root directory name and
        its parent name. Their joined form is ``<slide>_<region>``.
    include_transcripts
        Return projected lazy transcript points with known feature categories.
    include_cell_boundaries
        Return each cell's eager 2D XY footprint, the union of every valid
        source z-slice component. Holes and disconnected components survive.
    include_table
        Return eager canonical CSR expression and sparse blank measurements.
        Tables link to returned boundaries; excluding boundaries yields an
        explicitly unlinked table.
    include_images
        Return one lazy CYX image per selected z index, preserving source dtype.
    scale_factors
        Optional integer factors greater than one for generated image pyramid
        levels. None keeps the full-resolution single scale.
    chunks
        Positive integer or three positive CYX sizes. None preserves source
        chunks. Requested rechunking and pyramid generation remain lazy.

    Returns
    -------
    spatialdata.SpatialData
        Requested ``<slide>_<region>_z<index>`` images,
        ``<slide>_<region>_transcripts`` points,
        ``<slide>_<region>_polygons`` shapes and ``table``.
        Table observations follow count rows, features follow count columns,
        and metadata align by exact cell identity. Controls named
        ``Blank-<positive integer>`` reside in CSR ``obsm['blank']`` with names
        in ``uns['blank_names']``; other features remain expression.

    Raises
    ------
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        Requested artifacts are missing or have the wrong filesystem kind.
    ValueError
        Layouts, schemas, identities, geometry, calibration or values violate
        the supported export contract.
    TypeError
        Public options have invalid types.

    Notes
    -----
    The ``global`` coordinate system is mosaic pixels. Images use identity;
    source micron XY coordinates on shapes and points carry the forward
    micron-to-mosaic affine. Table ``obsm['spatial']`` retains micron XY
    centroids. Image layers and transcript ``z_raw`` are stack indices, not
    physical Z coordinates. Footprints discard z occupancy and are independent
    of image-layer selection; they are not a 3D segmentation.

    VPT's reassigned transcript CSV, when present, uses its count header and
    ``cell_id`` annotations. Otherwise the original instrument transcript CSV
    and original count header are required, and assignments are named
    ``original_cell_id`` without claiming correspondence to VPT boundaries.
    Unassigned ``-1`` events are retained. Unknown features and malformed rows
    fail when their partition is computed. Current cell assignments are checked
    against returned cell populations when those are already materialized.

    Source artifacts remain necessary for later raster and transcript
    computation. Counts use bounded CSV working batches; shapes and cell
    metadata are eager. Returned containers are owned, caller options are not
    mutated, and construction executes no raster or transcript tasks. See the
    MERSCOPE guide for complete schemas and deliberate API changes.
    """
    selection = make_selection(
        include_images=include_images,
        include_cell_boundaries=include_cell_boundaries,
        include_table=include_table,
        include_transcripts=include_transcripts,
        z_layers=z_layers,
    )
    options = make_raster_options(scale_factors, chunks)
    layout = discover_layout(
        path, selection, vpt_directory=vpt_directory, region_name=region_name, slide_name=slide_name
    )
    dataset_id = layout.names.dataset_id

    images = read_images(layout.images, dataset_id=dataset_id, options=options)

    table_data = None if layout.tables is None else read_table(layout.tables.counts, layout.tables.metadata)

    shapes: dict[str, GeoDataFrame] = {}
    points: dict[str, DataFrame] = {}
    tables: dict[str, AnnData] = {}
    boundaries = None

    if layout.physical is not None:
        transform = read_transform(layout.physical.transform)
        if layout.physical.boundaries is not None:
            boundaries = read_boundaries(layout.physical.boundaries.path, transform=transform)
            shapes[f"{dataset_id}_polygons"] = boundaries

        if layout.physical.transcripts is not None:
            source = layout.physical.transcripts
            if table_data is not None and layout.tables is not None and layout.tables.counts == source.vocabulary:
                features = table_data[1]
            else:
                features = read_feature_names(source.vocabulary)

            # Both private producers establish exact textual identities. Freeze
            # the available population once for deferred assignment validation.
            allowed_ids: tuple[str, ...] | None = None
            if source.assignment == "current":
                if boundaries is not None:
                    allowed_ids = cast("tuple[str, ...]", tuple(boundaries.index))
                elif table_data is not None:
                    allowed_ids = tuple(table_data[0].obs_names)

            points[f"{dataset_id}_transcripts"] = read_transcripts(
                source, features=features, transform=transform, allowed_cell_ids=allowed_ids
            )

    if table_data is not None:
        tables["table"] = finalize_table(
            table_data[0],
            region_name=layout.names.region,
            slide_name=layout.names.slide,
            dataset_id=dataset_id,
            target=f"{dataset_id}_polygons" if boundaries is not None else None,
            expected_instances=None if boundaries is None else boundaries.index,
        )

    result = SpatialData(images=images, points=points, shapes=shapes, tables=tables)
    set_reader_provenance(result.attrs, reader="merscope")
    return result
