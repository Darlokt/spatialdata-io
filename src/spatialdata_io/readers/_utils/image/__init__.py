"""Lazy, model-independent raster I/O for SpatialData readers.

TIFF-family sources use task-local ``tifffile`` Zarr stores and native storage
chunks, enabling Dask to cull reads outside a computed selection. PNG and JPEG
sources use delayed whole-frame decoding because those codecs do not expose a
bounded spatial-selection interface here. Vendor layout, axes semantics,
stacking, transformations, and SpatialData model construction remain reader
responsibilities, except the shared pyramid and chunk options with their
central defaults, the validation of caller ``data_axes`` and TIFF indexes,
the single-series rule, the agreement of two pixel-size declarations, the
shared rule labelling channels from a declared colour model or OME metadata,
and the normalization of a 2D image to ``c, y, x`` from its declared axes or
a caller override.
"""

from __future__ import annotations

from spatialdata_io.readers._utils.image._axes import normalize_raster_axes, validate_data_axes
from spatialdata_io.readers._utils.image._channels import channel_source, raster_channel_names
from spatialdata_io.readers._utils.image._cyx import normalize_cyx_image
from spatialdata_io.readers._utils.image._jpeg import read_jpeg
from spatialdata_io.readers._utils.image._ome import (
    ome_channel_names,
    ome_pixel_sizes_um,
    require_matching_pixel_sizes,
)
from spatialdata_io.readers._utils.image._options import RasterOptions, make_raster_options
from spatialdata_io.readers._utils.image._png import read_png
from spatialdata_io.readers._utils.image._types import LazyRaster
from spatialdata_io.readers._utils.image.tiff import (
    TiffMetadata,
    inspect_tiff,
    read_tiff,
    select_tiff_series,
    validate_tiff_index,
)

__all__ = [
    "LazyRaster",
    "RasterOptions",
    "TiffMetadata",
    "channel_source",
    "inspect_tiff",
    "make_raster_options",
    "normalize_cyx_image",
    "normalize_raster_axes",
    "ome_channel_names",
    "ome_pixel_sizes_um",
    "raster_channel_names",
    "read_jpeg",
    "read_png",
    "read_tiff",
    "require_matching_pixel_sizes",
    "select_tiff_series",
    "validate_data_axes",
    "validate_tiff_index",
]
