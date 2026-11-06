"""ISS explicit-artifact discovery and caller path policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.experimental.iss._layout import discover_layout, validate_artifact_names
from spatialdata_io.readers._utils.errors import ReaderFormatError
from tests.integration.experimental.iss.conftest import write_iss_dataset

if TYPE_CHECKING:
    from pathlib import Path


class TestDiscoverLayout:
    """Resolve each required artifact once with typed immutable paths."""

    def test_relative_files_and_immutable_layout(self, tmp_path: Path) -> None:
        write_iss_dataset(tmp_path)

        layout = discover_layout(
            tmp_path, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad"
        )

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
            discover_layout(tmp_path, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad")

    def test_absolute_override_may_be_outside_root(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        write_iss_dataset(root)
        outside = tmp_path / "outside.tiff"
        (root / "raw.tiff").rename(outside)

        layout = discover_layout(root, raw_image_file=outside, labels_file="labels.tiff", table_file="cells.h5ad")

        assert layout.raw_image == outside.resolve()

        with pytest.raises(ReaderFormatError, match="outside"):
            discover_layout(root, raw_image_file="../outside.tiff", labels_file="labels.tiff", table_file="cells.h5ad")

    def test_relative_symlink_cannot_escape(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        write_iss_dataset(root)
        outside = tmp_path / "outside.tiff"
        (root / "raw.tiff").rename(outside)
        (root / "raw.tiff").symlink_to(outside)

        with pytest.raises(ReaderFormatError, match="outside"):
            discover_layout(root, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad")

    @pytest.mark.parametrize("exists", [False, True])
    def test_root_kind(self, tmp_path: Path, *, exists: bool) -> None:
        root = tmp_path / "invalid"
        if exists:
            root.touch()

        with pytest.raises(NotADirectoryError if exists else FileNotFoundError):
            discover_layout(root, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad")


class TestValidateArtifactNames:
    """Check the caller's file suffixes without touching the filesystem."""

    @pytest.mark.parametrize(
        ("target", "option"), [("raw", "raw_image_file"), ("labels", "labels_file"), ("table", "table_file")]
    )
    def test_unsupported_suffix(self, target: str, option: str) -> None:
        with pytest.raises(ValueError, match=f"ISS {option} 'wrong.csv'"):
            validate_artifact_names(
                "wrong.csv" if target == "raw" else "raw.tiff",
                "wrong.csv" if target == "labels" else "labels.tiff",
                "wrong.csv" if target == "table" else "cells.h5ad",
            )

    @pytest.mark.parametrize("raster", ["raw.TIF", "raw.tiff", "dir/raw.btf", "/absent/raw.Tiff"])
    def test_supported_suffixes_are_case_insensitive(self, raster: str) -> None:
        validate_artifact_names(raster, raster, "cells.H5AD")
