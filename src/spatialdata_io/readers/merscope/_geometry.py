"""Project valid micron-space cell slices to complete two-dimensional footprints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import shapely
from spatialdata.models import ShapesModel

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers.merscope._schemas import STRING_DTYPE, boundary_columns, normalize_boundary_ids

if TYPE_CHECKING:
    from spatialdata.transformations import Affine

    from spatialdata_io.readers._utils.paths import ArtifactFile

_SPATIAL_DIMENSIONS = 2


def read_boundaries(path: ArtifactFile, *, transform: Affine) -> gpd.GeoDataFrame:
    """Union every valid component of each cell across z without repairing geometry."""
    context = ReaderErrorContext("merscope", "cell boundaries")
    try:
        with pq.ParquetFile(path) as source:
            columns = boundary_columns(source.schema_arrow.names, path)
            table = source.read(columns=list(columns))
    except (OSError, ValueError) as error:
        add_reader_context(error, context=context, path=path)
        raise
    identifiers = normalize_boundary_ids(table, path)
    try:
        geometry = shapely.from_wkb(table["Geometry"].to_numpy())
    except (TypeError, shapely.errors.GEOSException) as error:
        raise ReaderFormatError(
            context=context, path=path, found="malformed WKB geometry", expected="valid planar polygonal WKB slices"
        ) from error
    del table
    geometry_types = shapely.get_type_id(geometry)
    if not (
        (~shapely.is_missing(geometry)).all()
        and (~shapely.is_empty(geometry)).all()
        and shapely.is_valid(geometry).all()
        and (shapely.get_coordinate_dimension(geometry) == _SPATIAL_DIMENSIONS).all()
        and (
            (geometry_types == shapely.GeometryType.POLYGON) | (geometry_types == shapely.GeometryType.MULTIPOLYGON)
        ).all()
    ):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="missing, empty, invalid or nonplanar geometry",
            expected="valid finite 2D Polygon or MultiPolygon slices",
        )
    # Only identity and geometry are entity-wide. Z-specific row annotations
    # cannot truthfully describe the all-plane footprint and are not aggregated.
    # Grouping the explicitly typed identity column avoids index-level
    # aggregation overhead while preserving the first source occurrence.
    frame = gpd.GeoDataFrame({"EntityID": pd.Series(identifiers, dtype=STRING_DTYPE)}, geometry=geometry)
    try:
        footprints = frame.dissolve(by="EntityID", sort=False)
    except shapely.errors.GEOSException as error:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="footprint union failed",
            expected="a valid 2D footprint for each source cell",
        ) from error
    if footprints.empty or not footprints.geometry.is_valid.all():
        raise ReaderFormatError(
            context=context,
            path=path,
            found="empty or invalid projected footprints",
            expected="one valid footprint per source cell",
        )
    return ShapesModel.parse(footprints, transformations={"global": transform})
