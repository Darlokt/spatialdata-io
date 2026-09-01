"""Test the intentionally experimental ISS public surface."""

from __future__ import annotations

from pathlib import Path

import spatialdata_io
from spatialdata_io import experimental
from spatialdata_io.readers.iss import iss


class TestIssExports:
    """Keep ISS experimental until its Stage B cohort passes."""

    def test_is_experimental_and_not_a_root_export(self) -> None:
        assert "iss" not in spatialdata_io.__all__
        assert "iss" not in dir(spatialdata_io)
        assert not hasattr(spatialdata_io, "iss")
        assert experimental.iss is iss

    def test_api_documentation_lists_only_experimental_export(self) -> None:
        api_page = Path(__file__).parents[2] / "docs" / "api.md"
        contents = api_page.read_text(encoding="utf-8")

        assert "    experimental.iss\n" in contents
        assert "\n    iss\n" not in contents
