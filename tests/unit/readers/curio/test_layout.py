"""Curio sample discovery and artifact containment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.curio._layout import _discover_layout
from tests.integration.readers.curio._fixtures import write_curio

if TYPE_CHECKING:
    from pathlib import Path


class TestDiscoverLayout:
    """Discover one complete sample deterministically."""

    def test_exact_suffixes_and_immutable_paths(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        (tmp_path / "old_anndata.h5ad.backup").touch()
        layout = _discover_layout(tmp_path)
        assert layout.root == tmp_path.resolve()
        assert layout.counts == (tmp_path / "sample_anndata.h5ad").resolve()
        assert not hasattr(layout, "__dict__")

    @pytest.mark.parametrize(
        "suffix",
        [
            "anndata.h5ad",
            "cluster_assignment.txt",
            "Metrics.csv",
            "variable_features_clusters.txt",
            "variable_features_spatial_moransi.txt",
        ],
    )
    def test_missing_artifact(self, tmp_path: Path, suffix: str) -> None:
        write_curio(tmp_path)
        (tmp_path / f"sample_{suffix}").unlink()
        with pytest.raises(FileNotFoundError, match="curio"):
            _discover_layout(tmp_path)

    def test_ambiguity(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        (tmp_path / "second_anndata.h5ad").touch()
        with pytest.raises(ReaderFormatError, match="2 candidates"):
            _discover_layout(tmp_path)

    def test_conflicting_prefix(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        (tmp_path / "sample_Metrics.csv").rename(tmp_path / "other_Metrics.csv")
        with pytest.raises(ReaderFormatError, match="prefix"):
            _discover_layout(tmp_path)

    def test_empty_prefix(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        for path in tmp_path.iterdir():
            path.rename(tmp_path / path.name.removeprefix("sample"))
        with pytest.raises(ReaderFormatError, match="prefix"):
            _discover_layout(tmp_path)

    def test_artifact_directory(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        path = tmp_path / "sample_Metrics.csv"
        path.unlink()
        path.mkdir()
        with pytest.raises(IsADirectoryError):
            _discover_layout(tmp_path)

    def test_artifact_escape(self, tmp_path: Path) -> None:
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        write_curio(dataset)
        artifact = dataset / "sample_Metrics.csv"
        outside = tmp_path / "metrics.csv"
        artifact.rename(outside)
        artifact.symlink_to(outside)
        with pytest.raises(ReaderFormatError, match="outside"):
            _discover_layout(dataset)
