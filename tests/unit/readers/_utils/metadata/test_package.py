"""Tests for the narrow structured-metadata package surface."""

from __future__ import annotations

from spatialdata_io.readers import _utils
from spatialdata_io.readers._utils import metadata
from spatialdata_io.readers._utils.metadata import parse_json_object, read_json_object


class TestMetadataPackage:
    def test_exports_only_json_object_parsers(self) -> None:
        assert metadata.__all__ == ["parse_json_object", "read_json_object"]
        assert metadata.read_json_object is read_json_object
        assert metadata.parse_json_object is parse_json_object
        assert not hasattr(_utils, "read_json_object")
