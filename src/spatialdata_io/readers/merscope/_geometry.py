"""Build one 2D footprint per cell from the per-slice boundary Parquet.

Instrument ``cell_boundaries.parquet`` and VPT ``*_micron_space.parquet`` list
one WKB Polygon or MultiPolygon per cell and z slice, in micrometres. A cell's
footprint is the union of all its slices, so cells absent from z0, holes and
disconnected components survive; it is an XY projection, not a 3D
segmentation, and does not depend on the selected image planes.

Invalid slices follow the invalid-geometry policy per cell: ``"keep"`` returns
the literal components of every slice of that cell as one MultiPolygon
(``geometry_representation == "source_components"``) instead of a union, which
could repair or change it; ``"drop"`` removes the cell; ``"raise"`` fails.
Vertices are never changed. Missing, empty, nonfinite or non-polygonal slices
fail under every policy. Footprints are in the micrometre ``global`` frame
(identity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from shapely.geometry import MultiPolygon
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.geometry import invalid_geometry_mask, raise_invalid_geometries
from spatialdata_io.readers.merscope._schemas import (
    BOUNDARIES,
    GEOMETRY_REPRESENTATION_DTYPE,
    STRING_DTYPE,
    boundary_columns,
    normalize_boundary_ids,
    schema_error,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.geometry import InvalidGeometryPolicy
    from spatialdata_io.readers._utils.paths import ArtifactFile

_SPATIAL_DIMENSIONS = 2
_BOUNDARY_BATCH_ROWS = 4096
_CONTEXT = ReaderErrorContext("merscope", BOUNDARIES)


@dataclass(frozen=True, slots=True)
class CellBoundaries:
    """Returned footprints and the complete source population, including dropped cells."""

    shapes: gpd.GeoDataFrame
    source_ids: pd.Index


def _read_slices(path: ArtifactFile) -> tuple[pd.Index, NDArray[np.object_]]:
    """Decode projected batches into the required eager geometry in source row order."""
    identifiers: list[pd.Index] = []
    try:
        with pq.ParquetFile(path) as source:
            columns = boundary_columns(source.schema_arrow.names, path)
            if not source.metadata.num_rows:
                raise schema_error(
                    path, BOUNDARIES, found="no cell boundaries", expected="a nonempty cell footprint population"
                )
            geometry = np.empty(source.metadata.num_rows, dtype=object)
            start = 0
            for batch in source.iter_batches(batch_size=_BOUNDARY_BATCH_ROWS, columns=list(columns)):
                identifiers.append(normalize_boundary_ids(batch, path))
                try:
                    slices = shapely.from_wkb(batch.column("Geometry").to_numpy(zero_copy_only=False))
                except (TypeError, shapely.errors.GEOSException) as error:
                    raise ReaderFormatError(
                        context=_CONTEXT,
                        path=path,
                        found="malformed WKB geometry",
                        expected="finite planar polygonal WKB slices",
                    ) from error

                # WKB and coordinate-validation temporaries live for one batch;
                # final geometry remains eager for the whole-cell policy and union.
                types = shapely.get_type_id(slices)
                if not (
                    (~shapely.is_missing(slices)).all()
                    and (~shapely.is_empty(slices)).all()
                    and (shapely.get_coordinate_dimension(slices) == _SPATIAL_DIMENSIONS).all()
                    and ((types == shapely.GeometryType.POLYGON) | (types == shapely.GeometryType.MULTIPOLYGON)).all()
                    and np.isfinite(shapely.get_coordinates(slices)).all()
                ):
                    raise ReaderFormatError(
                        context=_CONTEXT,
                        path=path,
                        found="missing, empty, nonfinite or nonplanar geometry",
                        expected="nonempty finite 2D Polygon or MultiPolygon slices",
                    )
                stop = start + batch.num_rows
                geometry[start:stop] = slices
                start = stop
    except (OSError, ValueError) as error:
        add_reader_context(error, context=_CONTEXT, path=path)
        raise
    return identifiers[0].append(identifiers[1:]), geometry


def _union_footprints(frame: gpd.GeoDataFrame, path: ArtifactFile) -> gpd.GeoDataFrame:
    """Union the slices of each cell in first-appearance order; the union must be valid."""
    try:
        footprints = frame.dissolve(by="EntityID", sort=False)
    except shapely.errors.GEOSException as error:
        raise ReaderFormatError(
            context=_CONTEXT,
            path=path,
            found="footprint union failed",
            expected="a valid nonempty 2D footprint per source cell",
        ) from error
    if (
        footprints.geometry.is_empty.any()
        or not footprints.geometry.is_valid.all()
        or not footprints.geom_type.isin(("Polygon", "MultiPolygon")).all()
    ):
        raise ReaderFormatError(
            context=_CONTEXT,
            path=path,
            found="empty or invalid projected footprints",
            expected="a valid nonempty polygonal union per source cell",
        )
    footprints["geometry_representation"] = pd.Series(
        "union_footprint", index=footprints.index, dtype=GEOMETRY_REPRESENTATION_DTYPE
    )
    return footprints


def _source_components(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Collect every polygon of each cell's slices into one MultiPolygon."""
    # Construct collections, never a topology operation: preserve every source
    # row/component, including overlapping or invalid rings, in source order.
    groups = frame.groupby("EntityID", sort=False).indices
    geometry = frame.geometry.to_numpy()
    components = [MultiPolygon(shapely.get_parts(geometry[rows]).tolist()) for rows in groups.values()]
    # Match the union index dtype so the output does not depend on which cells are invalid.
    index = pd.Index(list(groups), dtype=STRING_DTYPE, name="EntityID")
    result = gpd.GeoDataFrame(geometry=components, index=index)
    result["geometry_representation"] = pd.Series("source_components", index=index, dtype=GEOMETRY_REPRESENTATION_DTYPE)
    return result


def read_boundaries(path: ArtifactFile, *, policy: InvalidGeometryPolicy) -> CellBoundaries:
    """Union valid-cell slices and apply whole-cell invalid-geometry semantics."""
    identifiers, geometry = _read_slices(path)
    source_ids = identifiers.unique()
    invalid = invalid_geometry_mask(geometry)
    invalid_ids = identifiers[invalid].unique()
    invalid_cells = source_ids.isin(invalid_ids)
    if policy == "raise":
        raise_invalid_geometries(invalid_cells, source_ids.tolist(), context=_CONTEXT, path=path)
    retained_ids = source_ids[~invalid_cells] if policy == "drop" else source_ids
    if retained_ids.empty:
        raise ReaderFormatError(
            context=_CONTEXT,
            path=path,
            found="all cell boundaries dropped, leaving an empty element",
            expected="at least one retained cell boundary",
        )
    invalid_rows = identifiers.isin(invalid_ids)
    slices = gpd.GeoDataFrame({"EntityID": pd.Series(identifiers, dtype=STRING_DTYPE)}, geometry=geometry)
    parts: list[gpd.GeoDataFrame] = []
    if (~invalid_rows).any():
        parts.append(_union_footprints(slices.loc[~invalid_rows] if invalid_rows.any() else slices, path))
    if policy == "keep" and invalid_rows.any():
        parts.append(_source_components(slices.loc[invalid_rows]))
    footprints = parts[0] if len(parts) == 1 else gpd.GeoDataFrame(pd.concat(parts), geometry="geometry")
    if not footprints.index.equals(retained_ids):
        footprints = footprints.loc[retained_ids]
    return CellBoundaries(ShapesModel.parse(footprints, transformations={"global": Identity()}), source_ids)
