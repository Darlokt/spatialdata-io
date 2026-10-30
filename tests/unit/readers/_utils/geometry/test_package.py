"""Tests for the narrow invalid-geometry package surface."""

from __future__ import annotations

from spatialdata_io.readers import _utils
from spatialdata_io.readers._utils import geometry
from spatialdata_io.readers._utils.geometry import (
    geojson_polygons,
    invalid_geometry_mask,
    raise_invalid_geometries,
    validate_invalid_geometry_policy,
)


class TestGeometryPackage:
    def test_exports_only_the_geometry_boundary(self) -> None:
        assert geometry.__all__ == [
            "InvalidGeometryPolicy",
            "geojson_polygons",
            "invalid_geometry_mask",
            "raise_invalid_geometries",
            "validate_invalid_geometry_policy",
        ]
        assert geometry.geojson_polygons is geojson_polygons
        assert geometry.invalid_geometry_mask is invalid_geometry_mask
        assert geometry.raise_invalid_geometries is raise_invalid_geometries
        assert geometry.validate_invalid_geometry_policy is validate_invalid_geometry_policy
        assert not hasattr(_utils, "invalid_geometry_mask")
