"""Tests for the narrow typed count-source package surface."""

from __future__ import annotations

from spatialdata_io.readers._utils import counts


class TestCountsPackage:
    def test_exports_only_typed_10x_sources(self) -> None:
        assert counts.__all__ == ["read_10x_h5", "read_10x_mtx"]
