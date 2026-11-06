"""Lazy, model-independent raster I/O for SpatialData readers.

TIFF-family sources use task-local ``tifffile`` Zarr stores and native storage
chunks, enabling Dask to cull reads outside a computed selection. PNG and JPEG
sources use delayed whole-frame decoding because those codecs do not expose a
bounded spatial-selection interface here. Vendor layout, axes semantics,
stacking, transformations, and SpatialData model construction remain reader
responsibilities, except the shared pyramid and chunk options with their
central defaults and the shared rule labelling channels from a declared
colour model or OME metadata.
"""

from __future__ import annotations

from spatialdata_io.readers._utils.image._axes import normalize_raster_axes
from spatialdata_io.readers._utils.image._channels import raster_channel_names
from spatialdata_io.readers._utils.image._jpeg import read_jpeg
from spatialdata_io.readers._utils.image._ome import ome_channel_names, ome_pixel_sizes_um
from spatialdata_io.readers._utils.image._options import RasterOptions, make_raster_options
from spatialdata_io.readers._utils.image._png import read_png
from spatialdata_io.readers._utils.image._types import LazyRaster
from spatialdata_io.readers._utils.image.tiff import TiffMetadata, inspect_tiff, read_tiff

__all__ = [
    "LazyRaster",
    "RasterOptions",
    "TiffMetadata",
    "inspect_tiff",
    "make_raster_options",
    "normalize_raster_axes",
    "ome_channel_names",
    "ome_pixel_sizes_um",
    "raster_channel_names",
    "read_jpeg",
    "read_png",
    "read_tiff",
]
