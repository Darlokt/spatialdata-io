"""Lazy, model-independent raster I/O for SpatialData readers.

TIFF-family sources use task-local ``tifffile`` Zarr stores and native storage
chunks, enabling Dask to cull reads outside a computed selection. PNG and JPEG
sources use delayed whole-frame decoding because those codecs do not expose a
bounded spatial-selection interface here. Vendor layout, axes semantics,
stacking, transformations, and SpatialData model construction remain reader
responsibilities.
"""

from __future__ import annotations

from spatialdata_io.readers._utils.image._axes import normalize_raster_axes
from spatialdata_io.readers._utils.image._jpeg import read_jpeg
from spatialdata_io.readers._utils.image._png import read_png
from spatialdata_io.readers._utils.image._types import LazyRaster
from spatialdata_io.readers._utils.image.tiff import TiffMetadata, inspect_tiff, read_tiff

__all__ = [
    "LazyRaster",
    "TiffMetadata",
    "inspect_tiff",
    "normalize_raster_axes",
    "read_jpeg",
    "read_png",
    "read_tiff",
]
