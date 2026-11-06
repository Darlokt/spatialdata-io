"""Generic readers for one caller-selected image or GeoJSON file."""

from __future__ import annotations

from spatialdata_io.readers.generic._images import image
from spatialdata_io.readers.generic._reader import generic
from spatialdata_io.readers.generic._shapes import geojson

__all__ = ["generic", "geojson", "image"]
