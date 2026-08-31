"""Tests for deterministic raster path validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import RasterFormatError
from spatialdata_io.readers._utils.image._path import normalize_file_path

if TYPE_CHECKING:
    from pathlib import Path


class TestNormalizeFilePath:
    """Resolve regular files without assigning format meaning to names."""

    def test_resolves_file_with_arbitrary_name(self, tmp_path: Path) -> None:
        path = tmp_path / "image.data"
        path.touch()

        result = normalize_file_path(path)

        assert result == path.resolve()

    def test_rejects_directory(self, tmp_path: Path) -> None:
        directory = tmp_path / "image.png"
        directory.mkdir()

        with pytest.raises(RasterFormatError, match="regular file"):
            normalize_file_path(directory)

    def test_propagates_missing_path_context(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.png"

        with pytest.raises(FileNotFoundError, match=r"missing\.png"):
            normalize_file_path(path)
