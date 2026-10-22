"""ISS explicit-artifact discovery and caller path policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.iss._layout import _discover_layout
from tests.integration.readers.iss.conftest import write_iss_dataset

if TYPE_CHECKING:
    from pathlib import Path


class TestDiscoverLayout:
    """Resolve each required artifact once with typed immutable paths."""

    def test_relative_files_and_immutable_layout(self, tmp_path: Path) -> None:
        write_iss_dataset(tmp_path)
        layout = _discover_layout(
            tmp_path, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad"
        )
        assert layout.root == tmp_path.resolve()
        assert layout.raw_image == (tmp_path / "raw.tiff").resolve()
        assert layout.labels == (tmp_path / "labels.tiff").resolve()
        assert layout.table == (tmp_path / "cells.h5ad").resolve()
        assert not hasattr(layout, "__dict__")

    @pytest.mark.parametrize("filename", ["raw.tiff", "labels.tiff", "cells.h5ad"])
    @pytest.mark.parametrize("wrong_kind", [False, True])
    def test_missing_and_directory_artifacts(self, tmp_path: Path, filename: str, *, wrong_kind: bool) -> None:
        write_iss_dataset(tmp_path)
        (tmp_path / filename).unlink()
        if wrong_kind:
            (tmp_path / filename).mkdir()
        with pytest.raises(IsADirectoryError if wrong_kind else FileNotFoundError):
            _discover_layout(tmp_path, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad")

    def test_absolute_override_may_be_outside_root(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        write_iss_dataset(root)
        outside = tmp_path / "outside.tiff"
        (root / "raw.tiff").rename(outside)
        layout = _discover_layout(root, raw_image_file=outside, labels_file="labels.tiff", table_file="cells.h5ad")
        assert layout.raw_image == outside.resolve()
        with pytest.raises(ReaderFormatError, match="outside"):
            _discover_layout(root, raw_image_file="../outside.tiff", labels_file="labels.tiff", table_file="cells.h5ad")

    def test_relative_symlink_cannot_escape(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        write_iss_dataset(root)
        outside = tmp_path / "outside.tiff"
        (root / "raw.tiff").rename(outside)
        (root / "raw.tiff").symlink_to(outside)
        with pytest.raises(ReaderFormatError, match="outside"):
            _discover_layout(root, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad")

    @pytest.mark.parametrize("target", ["raw", "labels", "table"])
    def test_unsupported_suffix(self, tmp_path: Path, target: str) -> None:
        write_iss_dataset(tmp_path)
        (tmp_path / "wrong.csv").touch()
        with pytest.raises(ValueError, match="ISS"):
            _discover_layout(
                tmp_path,
                raw_image_file="wrong.csv" if target == "raw" else "raw.tiff",
                labels_file="wrong.csv" if target == "labels" else "labels.tiff",
                table_file="wrong.csv" if target == "table" else "cells.h5ad",
            )

    @pytest.mark.parametrize("exists", [False, True])
    def test_root_kind(self, tmp_path: Path, *, exists: bool) -> None:
        root = tmp_path / "invalid"
        if exists:
            root.touch()
        with pytest.raises(NotADirectoryError if exists else FileNotFoundError):
            _discover_layout(root, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad")
