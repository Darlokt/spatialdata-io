"""Assemble the selected Spatial Genomics seqFISH ROIs into one SpatialData object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import pandas as pd
from spatialdata import SpatialData

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.seqfish._geometry import make_circles, read_boundaries
from spatialdata_io.readers.seqfish._layout import discover_layout, make_selection
from spatialdata_io.readers.seqfish._rasters import read_calibration, read_dapi, read_image, read_labels
from spatialdata_io.readers.seqfish._tables import finalize_table, read_table, select_cells
from spatialdata_io.readers.seqfish._transcripts import read_transcripts

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from anndata import AnnData
    from dask.dataframe import DataFrame
    from geopandas import GeoDataFrame
    from spatialdata.transformations import Scale
    from xarray import DataArray, DataTree


def seqfish(  # noqa: PLR0913, RUF100 - Supported public products remain explicit.
    path: str | Path,
    *,
    include_images: bool = True,
    include_labels: bool = True,
    include_transcripts: bool = True,
    include_shapes: bool = True,
    shape_geometry: Literal["polygons", "circles", "both"] = "polygons",
    invalid_geometries: Literal["keep", "drop", "raise"] = "keep",
    rois: Sequence[str] | None = None,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
    path_policy: Literal["strict", "relaxed"] = "strict",
) -> SpatialData:
    """Read the regions of interest (ROIs) of a Spatial Genomics seqFISH export.

    Supported layout (one flat directory; the export has no version marker)::

        dataset/
        ├── <ROI>_CellxGene.csv          # required: cell ID, then one count per feature
        ├── <ROI>_CellCoordinates.csv    # required: label, area, center_x, center_y
        ├── <ROI>_DAPI.tiff              # OME-TIFF with physical pixel sizes
        ├── <ROI>_Segmentation.tiff      # integer cell labels on the DAPI grid
        ├── <ROI>_TranscriptList.csv     # name, x, y, optional z and cell
        └── <ROI>_Boundaries.geojson     # cell polygons in DAPI pixels, Feature id = cell ID

    - ROIs are the prefixes of the ``CellxGene``/``CellCoordinates`` files.
      Every selected ROI needs both; the other files are required only by the
      products that read them. The DAPI file calibrates images, labels and
      polygons, so it is inspected even when ``include_images=False``.
    - The ROI prefix starts every element name. A prefix that is not a valid
      SpatialData element name is repaired with a warning (``Roi 1`` becomes
      ``Roi_1``); selected ROIs whose names then agree ignoring case fail.

    Parameters
    ----------
    path
        Export directory containing the ROI files.
    include_images
        Return the DAPI image of each ROI.
    include_labels
        Return the segmentation of each ROI and link its table to it.
    include_transcripts
        Return the transcripts of each ROI as points.
    include_shapes
        Return the cell shapes chosen by ``shape_geometry``.
    shape_geometry
        ``"polygons"`` (the GeoJSON boundaries), ``"circles"`` (centroid
        circles with the cell's area) or ``"both"``. Ignored without shapes.
    invalid_geometries
        Policy for polygons with invalid topology: ``"keep"`` them unchanged,
        ``"drop"`` the cell's polygon, circle and table row, or ``"raise"``.
        Boundary and table cell IDs must agree before anything is dropped.
        Labels and transcripts are never changed. Circles-only reads do not
        read the polygons.
    rois
        Exact ROI file prefixes to read, in this order. ``None`` reads every
        ROI in lexical prefix order; an empty sequence is invalid.
    scale_factors
        Integer factors of at least two, one per generated pyramid level of
        the images and labels; empty for a single scale. Label pyramids need
        labels of at most 32 bits.
    chunks
        Chunk size for every axis, or ``(c, y, x)`` sizes of the images;
        labels use the ``y`` and ``x`` sizes.
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
        Per selected ROI, with ``<ROI>`` its (repaired) name:

        - ``tables["table_<ROI>"]``: eager AnnData in count-row and
          feature-header order. ``X`` is a canonical CSR sparse array of the
          exact counts, float32 through 2**24, else float64 through 2**53.
          ``obs`` holds ``area`` (µm²) and the integer cell ID
          ``instance_id`` (uint64); ``obsm["spatial"]`` the float64 centroids (µm).
          Observation names are positional strings from ``"0"``; dropped
          cells leave gaps. With labels, categorical ``region`` links the
          table to ``<ROI>_Segmentation``; otherwise it is unlinked.
        - ``images["<ROI>_DAPI"]``: lazy ``(c, y, x)`` image in the source
          dtype, channels named by OME channel names or IDs.
        - ``labels["<ROI>_Segmentation"]``: lazy ``(y, x)`` integer labels in
          the source dtype. OME physical sizes the segmentation declares must
          equal the DAPI ones; a size it does not declare is inherited.
        - ``points["<ROI>_TranscriptList"]``: lazy two-dimensional points
          ``x``, ``y`` with categorical ``name`` (categories in feature-header
          order), optional ``cell`` (``uint64[pyarrow]``, ``0`` unassigned)
          and optional ``uint64[pyarrow]`` plane index ``z_raw``.
        - ``shapes["<ROI>_Boundaries"]``: polygons indexed by cell ID in
          table order, with the GeoJSON properties as columns.
        - ``shapes["<ROI>_CellCoordinates"]``: circles at the centroids with
          radius ``sqrt(area / pi)``, indexed by cell ID.

        ``attrs`` record the reader and package version.

    Raises
    ------
    TypeError
        If an option has the wrong type.
    ValueError
        If an option has an invalid value.
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If the directory or a required file is missing or has the wrong kind.
    ValueError
        A contextual ``ReaderFormatError`` naming the artifact and path if the
        layout, a schema or a value is malformed or files disagree: unknown
        ROIs, colliding ROI names, duplicate or mismatching cell IDs,
        malformed CSV records or values (see Notes), counts above 2**53,
        nonfinite coordinates, missing
        DAPI calibration, a segmentation off the DAPI grid or too narrow for
        the cell IDs. Transcript records are validated when their partition
        is computed.

    Warns
    -----
    UserWarning
        Once per ROI prefix that had to be repaired into an element name.

    Notes
    -----
    Coordinates: with one ROI every element is in ``global``, in
    micrometres; with several, each ROI keeps its own micrometre frame named
    after it, because ROIs are not registered to each other. Centroids,
    areas and transcript ``x``/``y`` are micrometres and use the identity.
    DAPI, segmentation and polygons are in DAPI pixels and use
    ``Scale([PhysicalSizeY, PhysicalSizeX])`` from the DAPI OME metadata
    (µm, nm or mm, converted to µm). Transcript ``z`` has no physical
    calibration. ``uns["spatialdata_io_positions"]`` describes the
    ``obsm["spatial"]`` frame.

    Laziness: tables and shapes are read eagerly; counts are parsed in
    bounded batches of about 1 MiB of text without a dense matrix, so a
    CellxGene record must be shorter than that. Pixels and transcript
    records are not read until computed, so keep the files available until
    the data are computed or written. The points' index is local to each
    partition.

    CSV values: Arrow parses every field into its declared type (cell IDs
    uint64, counts uint64, coordinates float64, transcript ``cell`` uint64,
    ``z`` uint64) and checks that every record has the header's width. Its
    grammar is the contract: spaces and tabs around numbers are trimmed,
    hexadecimal integers such as ``0x10`` are read, ``2.0`` or ``+1`` is not
    an integer, floats are correctly rounded (``1e-400`` reads as 0), and
    some malformed quoting is accepted. Read numeric fields cannot be empty;
    ignored annotation fields may be. Every transcript record occupies one
    line: a line with an odd number of double quotes fails when computed.
    """
    selection = make_selection(
        include_images=include_images,
        include_labels=include_labels,
        include_transcripts=include_transcripts,
        include_shapes=include_shapes,
        shape_geometry=shape_geometry,
        invalid_geometries=invalid_geometries,
        rois=rois,
    )
    options = make_raster_options(scale_factors=scale_factors, chunks=chunks)
    layouts = discover_layout(path, selection, path_policy=path_policy)

    images: dict[str, DataArray | DataTree] = {}
    labels: dict[str, DataArray | DataTree] = {}
    points: dict[str, DataFrame] = {}
    shapes: dict[str, GeoDataFrame] = {}
    tables: dict[str, AnnData] = {}
    for roi in layouts:
        frame = "global" if len(layouts) == 1 else roi.name
        table = read_table(roi.counts, roi.cells)
        # read_table returns an eager AnnData; Dataset2D is only a backed alternative.
        instance_ids = pd.Index(cast("pd.DataFrame", table.obs)["instance_id"])

        if roi.pixels is not None:
            scale: Scale
            if selection.images or selection.labels:
                dapi = read_dapi(roi.pixels.dapi)
                scale = dapi.scale
                if selection.images:
                    images[f"{roi.name}_DAPI"] = read_image(dapi, frame=frame, options=options)
                if roi.pixels.labels is not None:
                    labels[f"{roi.name}_Segmentation"] = read_labels(
                        roi.pixels.labels,
                        reference=dapi,
                        frame=frame,
                        options=options,
                        max_instance_id=int(instance_ids.max()),
                    )
            else:
                scale = read_calibration(roi.pixels.dapi)

            if roi.pixels.boundaries is not None:
                boundaries = read_boundaries(
                    roi.pixels.boundaries,
                    instance_ids=instance_ids,
                    frame=frame,
                    scale=scale,
                    policy=selection.invalid_geometries,
                )
                shapes[f"{roi.name}_Boundaries"] = boundaries
                if len(boundaries) != table.n_obs:
                    table = select_cells(table, boundaries.index, counts=roi.counts)

        if selection.shapes in {"circles", "both"}:
            shapes[f"{roi.name}_CellCoordinates"] = make_circles(table, frame=frame, cells=roi.cells)
        if roi.transcripts is not None:
            points[f"{roi.name}_TranscriptList"] = read_transcripts(
                roi.transcripts, features=tuple(table.var_names), frame=frame
            )
        tables[f"table_{roi.name}"] = finalize_table(
            table,
            frame=frame,
            region=f"{roi.name}_Segmentation" if selection.labels else None,
            counts=roi.counts,
        )

    result = SpatialData(images=images, labels=labels, points=points, shapes=shapes, tables=tables)
    set_reader_provenance(result.attrs, reader="seqfish")
    return result
