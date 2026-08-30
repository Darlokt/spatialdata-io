"""Tests for the narrow centralized raster package surface."""

from __future__ import annotations

from spatialdata_io.readers._utils import image


class TestImagePackage:
    """Keep reader-facing exports explicit and model-independent."""

    def test_exports_only_the_supported_internal_surface(self) -> None:
        assert image.__all__ == [
            "LazyRaster",
            "TiffMetadata",
            "inspect_tiff",
            "normalize_raster_axes",
            "read_jpeg",
            "read_png",
            "read_tiff",
        ]
