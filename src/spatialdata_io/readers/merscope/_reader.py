"""Read one Vizgen MERSCOPE region, optionally with a VPT re-segmentation, into SpatialData.

Boundaries, transcripts and cell centroids are natively in micrometres and
always use ``global`` with identity. Mosaics join ``global`` only through the
region's micron-to-mosaic calibration; without it they stay in
``mosaic_pixels``, so no registration is claimed that the data do not give.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from spatialdata import SpatialData

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.geometry import validate_invalid_geometry_policy
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.merscope._geometry import read_boundaries
from spatialdata_io.readers.merscope._images import read_images
from spatialdata_io.readers.merscope._layout import discover_layout, validate_options
from spatialdata_io.readers.merscope._tables import finalize_table, read_feature_names, read_table
from spatialdata_io.readers.merscope._transcripts import read_transcripts

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pandas as pd
    from anndata import AnnData
    from dask.dataframe import DataFrame
    from geopandas import GeoDataFrame

    from spatialdata_io.readers.merscope._geometry import CellBoundaries


def merscope(  # noqa: PLR0913, RUF100 - Explicit format choices keep the public boundary typed.
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
    invalid_geometries: Literal["keep", "drop", "raise"] = "keep",
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
    path_policy: Literal["strict", "relaxed"] = "strict",
) -> SpatialData:
    """Read a Vizgen MERSCOPE region and, optionally, its VPT re-segmentation.

    Supported layouts (trees in the MERSCOPE guide):

    - instrument output of one region: ``cell_by_gene.csv``,
      ``cell_metadata.csv``, ``cell_boundaries.parquet``,
      ``detected_transcripts.csv`` and ``images/`` with
      ``mosaic_<stain>_z<index>.tif`` and an optional
      ``micron_to_mosaic_pixel_transform.csv``;
    - Vizgen Post-processing Tool (VPT) output with Cellpose or Watershed
      segmentation in ``vpt_directory``: ``cell_by_gene.csv``,
      ``cell_metadata.csv``, exactly one ``cellpose_micron_space.parquet`` or
      ``watershed_micron_space.parquet``, and optionally reassigned
      ``detected_transcripts.csv``. Mosaics still come from the region.

    Only the artifacts of selected products are required. File names establish
    no software release, so no format version is detected.

    Parameters
    ----------
    path
        Region directory of the instrument output.
    vpt_directory
        VPT output directory whose cells replace the instrument cells. A
        relative path resolves against ``path``; an absolute path may be
        anywhere.
    z_layers
        Mosaic z indices in output order: unique nonnegative integers. ``None``
        reads every plane in ascending order. A scalar is not accepted.
    region_name, slide_name
        Parts of the element-name prefix ``<slide_name>_<region_name>``. They
        must be valid SpatialData element names. By default they are the
        region directory name and its parent directory name, repaired with a
        warning when they are not valid element names.
    include_transcripts
        Return the transcript points.
    include_cell_boundaries
        Return the cell footprints. The table links to them only when they
        are returned.
    include_table
        Return the cell table.
    include_images
        Return one image per selected z plane.
    invalid_geometries
        Cells with an invalid boundary slice: ``"keep"`` returns all their
        source polygons unchanged as one MultiPolygon; ``"drop"`` removes the
        cell from the footprints and its row from the table (transcript
        assignments are not rewritten); ``"raise"`` fails. Dropping every
        cell fails.
    scale_factors
        Integer factors of at least two, one per generated image pyramid
        level; empty for a single scale.
    chunks
        One positive size for every image axis, or ``(c, y, x)`` sizes.
    path_policy
        How symlinks inside the dataset directory are treated. ``"strict"``
        requires every discovered artifact and relative file option to resolve
        inside the directory. ``"relaxed"`` also accepts symlinks to targets
        stored elsewhere, as in git-annex or DVC datasets. Relative paths may
        not leave the directory with ``..`` in either mode, and absolute file
        options may point anywhere. File options keep the name the caller
        gave, and relaxed reads keep each artifact's own name rather than that
        of its link target.

    Returns
    -------
    spatialdata.SpatialData
        The selected elements, with ``<prefix>`` = ``<slide_name>_<region_name>``:

        - ``images["<prefix>_z<index>"]``: lazy ``(c, y, x)`` mosaic per
          selected plane in the source dtype; channels are the stains in
          lexical order. In ``global`` through the inverse of the
          micron-to-mosaic affine when that file exists, otherwise only in
          ``mosaic_pixels`` (identity).
        - ``points["<prefix>_transcripts"]``: lazy events with float64 ``x``,
          ``y`` (micrometres) and categorical ``gene`` in count-header order,
          plus, when present, int64 ``z_raw`` (stack index, not a physical
          depth), ``fov`` and ``barcode_id``, text ``transcript_id``, and the
          text assignment ``cell_id`` (instance key; ``-1`` unassigned) or
          ``original_cell_id`` (instrument events read beside VPT cells).
        - ``shapes["<prefix>_polygons"]``: one XY footprint per cell, the
          union of its z slices, indexed by text ``EntityID``, with a
          categorical ``geometry_representation`` (``union_footprint`` or
          ``source_components``). Not repaired.
        - ``tables["table"]``: rows in count-file order, features in
          count-header order. ``X`` is a CSR sparse array of gene counts,
          ``obsm["blank"]`` a CSR array of ``Blank-<n>`` controls named by
          ``uns["blank_names"]``; both float32 when every count is exact in
          float32, otherwise float64. ``obsm["spatial"]`` holds float64
          micrometre centroids; ``obs`` holds the cell metadata, text
          ``EntityID`` and categorical ``region``, ``slide`` and
          ``dataset_id``. Metadata column names that a SpatialData table
          does not accept (``cell type``) are sanitized with a warning
          (``cell_type``; a name that then collides, ignoring letter case,
          gets a ``_1``, ``_2``, ... suffix). Linked to the polygons through ``cells_region`` and
          ``EntityID`` when they are returned, otherwise unlinked.

        Points and shapes use ``global`` (micrometres) with identity; table
        centroids are in the same frame.

    Raises
    ------
    TypeError
        If an option has the wrong type.
    ValueError
        If an option has an invalid value, including names that are not valid
        element names.
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If a required artifact is missing or has the wrong kind.
    ReaderFormatError
        If an artifact is ambiguous or malformed, the count, metadata and
        boundary cell populations disagree, or a value is invalid (for
        example a count that is not exact in float64). Malformed transcript
        rows and unknown genes or assigned cells raise when their partition is
        computed.

    Warns
    -----
    UserWarning
        If a default name part taken from a directory name is repaired, or
        cell metadata column names are sanitized.

    Notes
    -----
    Building the result runs no pixel or transcript task, and the lazy
    elements keep reading the source files: keep them available until the
    data are computed or written. Counts, cell metadata and boundaries are
    read eagerly; counts in bounded batches. Current transcript assignments
    are checked against the boundary population (before ``"drop"``) or,
    without boundaries, the table. The returned containers are owned by the
    caller. See the MERSCOPE guide for the complete source schemas.
    """
    selection = validate_options(
        vpt_directory=vpt_directory,
        z_layers=z_layers,
        region_name=region_name,
        slide_name=slide_name,
        include_transcripts=include_transcripts,
        include_cell_boundaries=include_cell_boundaries,
        include_table=include_table,
        include_images=include_images,
    )
    policy = validate_invalid_geometry_policy(invalid_geometries)
    options = make_raster_options(scale_factors, chunks)
    layout = discover_layout(
        path,
        selection,
        vpt_directory=vpt_directory,
        region_name=region_name,
        slide_name=slide_name,
        path_policy=path_policy,
    )
    prefix = layout.names.dataset_id

    images = {} if layout.images is None else read_images(layout.images, dataset_id=prefix, options=options)
    table = None
    table_features: tuple[str, ...] = ()
    if layout.tables is not None:
        table, table_features = read_table(layout.tables)
    boundaries = None if layout.boundaries is None else read_boundaries(layout.boundaries, policy=policy)

    points: dict[str, DataFrame] = {}
    if layout.transcripts is not None:
        source = layout.transcripts
        shared_header = layout.tables is not None and layout.tables.counts == source.vocabulary
        points[f"{prefix}_transcripts"] = read_transcripts(
            source,
            features=table_features if shared_header else read_feature_names(source.vocabulary),
            cell_population=_current_cells(boundaries, table) if source.assignment == "current" else None,
        )

    shapes: dict[str, GeoDataFrame] = {} if boundaries is None else {f"{prefix}_polygons": boundaries.shapes}
    tables: dict[str, AnnData] = {}
    if table is not None:
        tables["table"] = finalize_table(table, names=layout.names, boundaries=boundaries)

    result = SpatialData(images=images, points=points, shapes=shapes, tables=tables)
    set_reader_provenance(result.attrs, reader="merscope")
    return result


def _current_cells(boundaries: CellBoundaries | None, table: AnnData | None) -> pd.Index | None:
    """Return the complete current cell population, before geometry filtering, when one is read."""
    if boundaries is not None:
        return boundaries.source_ids
    return None if table is None else table.obs_names
