"""Tests for the narrow typed count-source package surface."""

from __future__ import annotations

from spatialdata_io.readers._utils import counts


class TestCountsPackage:
    def test_exports_only_typed_10x_sources(self) -> None:
        assert counts.__all__ == [
            "MtxFiles",
            "read_10x_h5",
            "read_10x_h5_axes",
            "read_10x_h5_components",
            "read_10x_mtx",
            "read_10x_mtx_fields",
        ]
