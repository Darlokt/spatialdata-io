"""Shared invalid-geometry vocabulary, definition and error format for readers."""

from __future__ import annotations

from spatialdata_io.readers._utils.geometry._policy import (
    InvalidGeometryPolicy,
    invalid_geometry_mask,
    raise_invalid_geometries,
    validate_invalid_geometry_policy,
)

__all__ = [
    "InvalidGeometryPolicy",
    "invalid_geometry_mask",
    "raise_invalid_geometries",
    "validate_invalid_geometry_policy",
]
