"""Tests for the narrow provenance package surface."""

from __future__ import annotations

from spatialdata_io.readers._utils import provenance


class TestProvenancePackage:
    def test_exports_only_the_mapping_boundary(self) -> None:
        assert provenance.__all__ == ["set_reader_provenance"]
