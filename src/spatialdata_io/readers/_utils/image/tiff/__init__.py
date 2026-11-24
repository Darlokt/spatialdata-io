"""Inspect TIFF metadata and read a selected raster lazily.

Use ``inspect_tiff`` to resolve series, levels and optional physical page tags,
then pass the closed metadata and chosen selection to ``read_tiff``.
"""

from __future__ import annotations

from spatialdata_io.readers._utils.image.tiff._dask import read_tiff
from spatialdata_io.readers._utils.image.tiff._metadata import inspect_tiff, inspect_tiff_if_recognized
from spatialdata_io.readers._utils.image.tiff._selection import select_tiff_series, validate_tiff_index
from spatialdata_io.readers._utils.image.tiff._types import TiffMetadata, TiffPageMetadata

__all__ = [
    "TiffMetadata",
    "TiffPageMetadata",
    "inspect_tiff",
    "inspect_tiff_if_recognized",
    "read_tiff",
    "select_tiff_series",
    "validate_tiff_index",
]
