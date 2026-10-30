"""Shared GeoJSON polygon construction and invalid-geometry policy for readers."""

from __future__ import annotations

from spatialdata_io.readers._utils.geometry._geojson import geojson_polygons
from spatialdata_io.readers._utils.geometry._policy import (
    InvalidGeometryPolicy,
    invalid_geometry_mask,
    raise_invalid_geometries,
    validate_invalid_geometry_policy,
)

__all__ = [
    "InvalidGeometryPolicy",
    "geojson_polygons",
    "invalid_geometry_mask",
    "raise_invalid_geometries",
    "validate_invalid_geometry_policy",
]
