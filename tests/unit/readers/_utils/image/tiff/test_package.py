"""Tests for the TIFF subpackage surface."""

from __future__ import annotations

from spatialdata_io.readers._utils.image import tiff


class TestTiffPackage:
    """Expose inspection and explicit series-level reading only."""

    def test_exports_two_entry_points(self) -> None:
        assert tiff.__all__ == ["TiffMetadata", "inspect_tiff", "read_tiff"]
