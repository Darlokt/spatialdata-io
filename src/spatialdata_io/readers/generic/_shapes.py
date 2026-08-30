"""Generic GeoJSON loading and SpatialData shape-model construction."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

if TYPE_CHECKING:
    from geopandas import GeoDataFrame

VALID_SHAPE_TYPES = (".geojson",)


def geojson(path: str | Path, coordinate_system: str) -> GeoDataFrame:
    """Read GeoJSON polygons as a SpatialData shapes element.

    Parameters
    ----------
    path
        GeoJSON file containing polygons or multipolygons.
    coordinate_system
        Coordinate system receiving an identity transformation.

    Returns
    -------
    geopandas.GeoDataFrame
        Parsed SpatialData shapes element.
    """
    return ShapesModel.parse(Path(path), transformations={coordinate_system: Identity()})
