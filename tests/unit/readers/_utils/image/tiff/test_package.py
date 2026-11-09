"""Tests for the TIFF subpackage surface."""

from __future__ import annotations

from spatialdata_io.readers._utils.image import tiff


class TestTiffPackage:
    """Expose inspection, series selection and explicit series-level reading only."""

    def test_exports_the_entry_points(self) -> None:
        assert tiff.__all__ == [
            "TiffMetadata",
            "inspect_tiff",
            "read_tiff",
            "select_tiff_series",
            "validate_tiff_index",
        ]
