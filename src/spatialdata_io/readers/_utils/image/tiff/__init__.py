"""TIFF-family metadata inspection, series selection and native-chunk lazy reading."""

from __future__ import annotations

from spatialdata_io.readers._utils.image.tiff._dask import read_tiff
from spatialdata_io.readers._utils.image.tiff._metadata import inspect_tiff
from spatialdata_io.readers._utils.image.tiff._selection import select_tiff_series, validate_tiff_index
from spatialdata_io.readers._utils.image.tiff._types import TiffMetadata

__all__ = ["TiffMetadata", "inspect_tiff", "read_tiff", "select_tiff_series", "validate_tiff_index"]
