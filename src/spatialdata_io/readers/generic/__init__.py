"""Generic image and GeoJSON readers."""

from __future__ import annotations

from spatialdata_io.readers.generic._images import VALID_IMAGE_TYPES, image
from spatialdata_io.readers.generic._reader import generic
from spatialdata_io.readers.generic._shapes import VALID_SHAPE_TYPES, geojson

__all__ = ["VALID_IMAGE_TYPES", "VALID_SHAPE_TYPES", "generic", "geojson", "image"]
