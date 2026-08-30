"""TIFF-family metadata inspection and native-chunk lazy reading."""

from __future__ import annotations

from spatialdata_io.readers._utils.image.tiff._dask import read_tiff
from spatialdata_io.readers._utils.image.tiff._metadata import inspect_tiff
from spatialdata_io.readers._utils.image.tiff._types import TiffMetadata

__all__ = ["TiffMetadata", "inspect_tiff", "read_tiff"]
