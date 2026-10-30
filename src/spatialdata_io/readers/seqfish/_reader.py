"""Assemble explicitly calibrated, linked Spatial Genomics ROI elements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from spatialdata import SpatialData

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.geometry import validate_invalid_geometry_policy
from spatialdata_io.readers._utils.image import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS, make_raster_options
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.seqfish._geometry import make_circles, read_boundaries
from spatialdata_io.readers.seqfish._layout import discover_layout, make_selection
from spatialdata_io.readers.seqfish._rasters import read_dapi, read_image, read_labels
from spatialdata_io.readers.seqfish._tables import finalize_table, read_table
from spatialdata_io.readers.seqfish._transcripts import read_transcripts

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from anndata import AnnData
    from dask.dataframe import DataFrame
    from geopandas import GeoDataFrame
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
) -> SpatialData:
    """Read the repeated Spatial Genomics seqFISH ROI export layout.

    Each ROI requires ``<ROI>_CellxGene.csv`` and
    ``<ROI>_CellCoordinates.csv``. Requested products additionally require
    ``<ROI>_DAPI.tiff``, ``<ROI>_Segmentation.tiff``,
    ``<ROI>_Boundaries.geojson`` or ``<ROI>_TranscriptList.csv``.
    DAPI OME metadata calibrates every pixel-based product, even when the image
    itself is excluded. Only a single full-resolution series is supported.
    No instrument/software release is inferred from these filenames.

    Parameters
    ----------
    path
        Directory containing the ROI artifacts.
    include_images
        Return lazy DAPI images in CYX order, preserving dtype and OME channels.
    include_labels
        Return lazy YX integer segmentations and link tables to them. Excluding
        labels returns normalized unlinked tables without a substitute target.
    include_transcripts
        Return projected lazy transcript points. Values are validated when a
        partition is computed; feature names must belong to its CellxGene header.
    include_shapes
        Return the requested cell geometry representations.
    shape_geometry
        ``"polygons"`` uses explicit GeoJSON feature IDs in pixel coordinates;
        ``"circles"`` uses cell centroids and ``sqrt(area/pi)`` physical radii;
        ``"both"`` returns both. Ignored when shapes are excluded.
    invalid_geometries
        Keep invalid polygon topology unchanged, drop its cell's polygon,
        requested circle and table row, or raise. Source identity relationships
        are validated before dropping. Labels and transcripts remain unchanged.
        Circles-only or shapes-disabled reads never inspect polygons or drop cells.
    rois
        Unique ROI filename prefixes in desired order. None selects all in
        lexical order; an empty sequence is invalid.
    scale_factors
        Integer factors of at least two, one per generated pyramid level of the
        images and labels; empty for a single scale. Label pyramids require at
        most 32-bit labels to avoid upstream integer rounding.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes of the
        images; labels use the ``y`` and ``x`` sizes. By default one channel per
        chunk and square 1024-pixel tiles.

    Returns
    -------
    spatialdata.SpatialData
        Per-ROI elements ``table_<ROI>``, ``<ROI>_DAPI``,
        ``<ROI>_Segmentation``, ``<ROI>_TranscriptList``,
        ``<ROI>_Boundaries`` and/or ``<ROI>_CellCoordinates`` when requested.
        Tables contain eager finite canonical CSR count arrays in source feature
        and observation order; cells and boundaries align by exact top-level
        feature identifiers. GeoJSON properties are independent annotations.
        Polygon drops retain surviving positional row names, including gaps.

    Raises
    ------
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        Required artifacts are absent or have the wrong filesystem kind.
    ValueError
        Layout, schema, identities, geometry, calibration or numerical values
        violate the supported export contract.
    TypeError
        Public option types are invalid.

    Notes
    -----
    One selected ROI uses ``global`` in micrometers. Multiple independent ROIs
    retain their distinct ROI frame names. OME X/Y physical
    sizes calibrate pixels without exchanging axes. Cell and transcript XY
    coordinates are already physical; cell area is square micrometers. Source
    transcript z has no established physical calibration and is retained as
    ``z_raw`` slice annotation; points are spatially two-dimensional. Optional
    ``cell`` assignments retain zero for unassigned events. Point indexes are
    partition-local storage indexes, not globally unique event IDs.

    CSV inputs must be valid. Arrow checks record widths but does not enforce
    strict quoting; some malformed quote forms may be accepted. Numerical
    precision and exact integer-identifier checks apply to parsed values.

    Counts, cell metadata and shapes are eager and owned. CSV working batches
    are bounded; raster pixels and transcript partitions remain path-backed and
    lazy after return. Keep sources available for later computation. Caller
    selections are never mutated. See the seqFISH guide for the complete
    schemas, identifier policy and deliberate API changes.
    """
    policy = validate_invalid_geometry_policy(invalid_geometries)
    selection = make_selection(
        include_images=include_images,
        include_labels=include_labels,
        include_transcripts=include_transcripts,
        include_shapes=include_shapes,
        shape_geometry=shape_geometry,
        rois=rois,
    )
    options = make_raster_options(scale_factors=scale_factors, chunks=chunks)
    layout = discover_layout(path, selection)
    images: dict[str, DataArray | DataTree] = {}
    labels: dict[str, DataArray | DataTree] = {}
    points: dict[str, DataFrame] = {}
    shapes: dict[str, GeoDataFrame] = {}
    tables: dict[str, AnnData] = {}

    for roi in layout.rois:
        frame = "global" if len(layout.rois) == 1 else roi.name
        table = read_table(roi.counts, roi.cells)
        # read_table owns an eager AnnData; Dataset2D is only a backed alternative.
        obs = cast("pd.DataFrame", table.obs)

        if roi.pixels is not None:
            dapi = read_dapi(roi.pixels.dapi)
            if selection.images:
                images[f"{roi.name}_DAPI"] = read_image(dapi, roi=frame, options=options)
            if roi.pixels.labels is not None:
                element, dtype = read_labels(roi.pixels.labels, reference=dapi, roi=frame, options=options)
                if int(obs["instance_id"].max()) > np.iinfo(dtype.str).max:
                    raise ReaderFormatError(
                        context=ReaderErrorContext("seqfish", "label identity"),
                        path=roi.pixels.labels,
                        found="cell IDs exceeding the label dtype",
                        expected=f"IDs representable by {dtype}",
                    )
                labels[f"{roi.name}_Segmentation"] = element
            if roi.pixels.boundaries is not None:
                boundaries = read_boundaries(
                    roi.pixels.boundaries,
                    instance_ids=pd.Index(obs["instance_id"]),
                    roi=frame,
                    scale=dapi.scale,
                    policy=policy,
                )
                shapes[f"{roi.name}_Boundaries"] = boundaries
                if len(boundaries) != table.n_obs:
                    table = table[obs["instance_id"].isin(boundaries.index).to_numpy()].copy()

        if selection.shapes in {"circles", "both"}:
            shapes[f"{roi.name}_CellCoordinates"] = make_circles(table, roi=frame)

        if roi.transcripts is not None:
            points[f"{roi.name}_TranscriptList"] = read_transcripts(
                roi.transcripts, features=tuple(table.var_names), roi=frame
            )

        tables[f"table_{roi.name}"] = finalize_table(
            table,
            region=f"{roi.name}_Segmentation" if selection.labels else None,
            context=ReaderErrorContext("seqfish", f"{roi.name} table"),
        )

    result = SpatialData(images=images, labels=labels, points=points, shapes=shapes, tables=tables)
    set_reader_provenance(result.attrs, reader="seqfish")
    return result
