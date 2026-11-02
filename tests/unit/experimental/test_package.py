"""Test the experimental namespace and its ISS entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

import spatialdata_io
from spatialdata_io import experimental
from spatialdata_io.experimental.iss import iss


@pytest.mark.iss
class TestExperimentalPackage:
    """Expose ISS without root aliases or legacy converter re-exports."""

    def test_exports_only_iss(self) -> None:
        assert experimental.__all__ == ["iss"]
        assert experimental.iss is iss
        assert not hasattr(experimental, "from_legacy_anndata")
        assert not hasattr(experimental, "to_legacy_anndata")
        assert "iss" not in spatialdata_io.__all__
        assert "iss" not in dir(spatialdata_io)
        assert not hasattr(spatialdata_io, "iss")

    def test_api_documentation_lists_experimental_export(self) -> None:
        api_page = Path(__file__).parents[3] / "docs" / "api.md"
        contents = api_page.read_text(encoding="utf-8")

        assert "\n    experimental.iss\n" in contents
        assert "\n    iss\n" not in contents
