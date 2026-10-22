"""Test the supported ISS reader's root and package exports."""

from __future__ import annotations

from pathlib import Path

import spatialdata_io
from spatialdata_io import experimental
from spatialdata_io.readers.iss import iss


class TestIssExports:
    """Expose one supported ISS entry point through the lazy root API."""

    def test_is_a_root_export_without_an_experimental_alias(self) -> None:
        assert "iss" in spatialdata_io.__all__
        assert "iss" in dir(spatialdata_io)
        assert spatialdata_io.iss is iss
        assert "iss" not in experimental.__all__
        assert not hasattr(experimental, "iss")

    def test_api_documentation_lists_root_export(self) -> None:
        api_page = Path(__file__).parents[2] / "docs" / "api.md"
        contents = api_page.read_text(encoding="utf-8")

        assert "    experimental.iss\n" not in contents
        assert "\n    iss\n" in contents
