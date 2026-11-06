"""Curio sample discovery and artifact containment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.curio._layout import discover_layout
from tests.integration.readers.curio._fixtures import write_curio

if TYPE_CHECKING:
    from pathlib import Path


class TestDiscoverLayout:
    """Discover one complete sample deterministically."""

    def test_exact_suffixes_and_immutable_paths(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        (tmp_path / "old_anndata.h5ad.backup").touch()

        layout = discover_layout(tmp_path)

        assert layout.counts == (tmp_path / "sample_anndata.h5ad").resolve()
        assert not hasattr(layout, "__dict__")

    @pytest.mark.parametrize(
        ("suffix", "artifact"),
        [
            ("anndata.h5ad", "counts"),
            ("cluster_assignment.txt", "cluster assignments"),
            ("Metrics.csv", "metrics"),
            ("variable_features_clusters.txt", "cluster features"),
            ("variable_features_spatial_moransi.txt", "Moran features"),
        ],
    )
    def test_missing_artifact(self, tmp_path: Path, suffix: str, artifact: str) -> None:
        """Name the missing artifact and the file pattern that was searched for."""
        write_curio(tmp_path)
        (tmp_path / f"sample_{suffix}").unlink()

        with pytest.raises(FileNotFoundError, match="curio") as raised:
            discover_layout(tmp_path)

        assert f"{artifact} (*_{suffix})" in str(raised.value)

    def test_ambiguity(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        (tmp_path / "second_anndata.h5ad").touch()

        with pytest.raises(ReaderFormatError, match="2 candidates") as raised:
            discover_layout(tmp_path)

        assert raised.value.context.artifact == "counts (*_anndata.h5ad)"

    def test_conflicting_prefix(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        (tmp_path / "sample_Metrics.csv").rename(tmp_path / "other_Metrics.csv")

        with pytest.raises(ReaderFormatError, match="prefix") as raised:
            discover_layout(tmp_path)

        assert raised.value.context.artifact == "dataset layout"
        assert raised.value.path == tmp_path.resolve()

    def test_empty_prefix(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        for path in tmp_path.iterdir():
            path.rename(tmp_path / path.name.removeprefix("sample"))

        with pytest.raises(ReaderFormatError, match="prefix"):
            discover_layout(tmp_path)

    def test_artifact_directory(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        path = tmp_path / "sample_Metrics.csv"
        path.unlink()
        path.mkdir()

        with pytest.raises(IsADirectoryError):
            discover_layout(tmp_path)

    def test_artifact_escape(self, tmp_path: Path) -> None:
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        write_curio(dataset)
        artifact = dataset / "sample_Metrics.csv"
        outside = tmp_path / "metrics.csv"
        artifact.rename(outside)
        artifact.symlink_to(outside)

        with pytest.raises(ReaderFormatError, match="outside"):
            discover_layout(dataset)
