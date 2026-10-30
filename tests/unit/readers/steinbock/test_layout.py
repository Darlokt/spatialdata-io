"""Steinbock deterministic pairing, artifact validation, and containment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.steinbock._layout import _discover_layout

if TYPE_CHECKING:
    from pathlib import Path


def _write_paths(root: Path) -> None:
    """Create artifact paths without involving raster or table parsers."""
    (root / "ome").mkdir()
    (root / "cells.h5ad").touch()
    for kind in ("deepcell", "ilastik"):
        (root / f"masks_{kind}").mkdir()
        for name in ("sample-b", "sample-a"):
            (root / f"masks_{kind}" / f"{name}.tiff").touch()
    for name in ("sample-b", "sample-a"):
        (root / "ome" / f"{name}.ome.tiff").touch()


class TestDiscoverLayout:
    """Require a complete paired layout with contained validated artifacts."""

    @pytest.mark.parametrize("kind", ["deepcell", "ilastik"])
    def test_exact_pairing_and_deterministic_order(self, tmp_path: Path, kind: Literal["deepcell", "ilastik"]) -> None:
        _write_paths(tmp_path)
        (tmp_path / "ome" / "unrelated.ome.tiff.backup").touch()
        (tmp_path / "ome" / "unrelated.tiff").touch()
        (tmp_path / f"masks_{kind}" / "unrelated.tiff.backup").touch()

        layout = _discover_layout(tmp_path, labels_kind=kind)

        assert layout.root == tmp_path.resolve()
        assert layout.table == tmp_path / "cells.h5ad"
        assert [sample.name for sample in layout.samples] == ["sample-a", "sample-b"]
        assert not hasattr(layout, "__dict__")
        for sample in layout.samples:
            assert sample.image == tmp_path / "ome" / f"{sample.name}.ome.tiff"
            assert sample.labels == tmp_path / f"masks_{kind}" / f"{sample.name}.tiff"
            assert not hasattr(sample, "__dict__")

    @pytest.mark.parametrize("kind", ["file", "directory", "escape"])
    def test_optional_image_metadata_artifact(self, tmp_path: Path, kind: str) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _write_paths(root)
        path = root / "images.csv"
        if kind == "directory":
            path.mkdir()
            with pytest.raises(IsADirectoryError):
                _discover_layout(root, labels_kind="deepcell")
        elif kind == "escape":
            outside = tmp_path / "outside.csv"
            outside.touch()
            path.symlink_to(outside)
            with pytest.raises(ReaderFormatError, match="root"):
                _discover_layout(root, labels_kind="deepcell")
        else:
            path.touch()
            assert _discover_layout(root, labels_kind="deepcell").image_metadata == path

    def test_embedded_suffix_and_hyphens_preserved(self, tmp_path: Path) -> None:
        _write_paths(tmp_path)
        name = "sample.tiff-in-tissue"
        (tmp_path / "ome" / "sample-a.ome.tiff").rename(tmp_path / "ome" / f"{name}.ome.tiff")
        (tmp_path / "masks_deepcell" / "sample-a.tiff").rename(tmp_path / "masks_deepcell" / f"{name}.tiff")

        layout = _discover_layout(tmp_path, labels_kind="deepcell")

        assert [sample.name for sample in layout.samples] == ["sample-b", name]

    @pytest.mark.parametrize("name", ["sample name", "sample:name", "sample?", "sample[1]", "__private", ".", ".."])
    def test_invalid_sample_names(self, tmp_path: Path, name: str) -> None:
        _write_paths(tmp_path)
        (tmp_path / "ome" / f"{name}.ome.tiff").touch()
        (tmp_path / "masks_deepcell" / f"{name}.tiff").touch()

        with pytest.raises(ReaderFormatError, match="sample names") as caught:
            _discover_layout(tmp_path, labels_kind="deepcell")

        assert caught.value.context.reader == "steinbock"
        assert caught.value.path == tmp_path

    def test_case_insensitive_sample_collision(self, tmp_path: Path) -> None:
        _write_paths(tmp_path)
        (tmp_path / "ome" / "SAMPLE-A.ome.tiff").touch()
        (tmp_path / "masks_deepcell" / "SAMPLE-A.tiff").touch()

        with pytest.raises(ReaderFormatError, match="case-insensitively conflicting"):
            _discover_layout(tmp_path, labels_kind="deepcell")

    def test_invalid_kind(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="labels_kind"):
            _discover_layout(tmp_path, labels_kind=cast("Literal['deepcell', 'ilastik']", "other"))

    @pytest.mark.parametrize("missing", ["cells.h5ad", "ome", "masks_deepcell"])
    def test_missing_required_artifact(self, tmp_path: Path, missing: str) -> None:
        _write_paths(tmp_path)
        artifact = tmp_path / missing
        if artifact.is_dir():
            for child in artifact.iterdir():
                child.unlink()
            artifact.rmdir()
        else:
            artifact.unlink()

        with pytest.raises(FileNotFoundError) as raised:
            _discover_layout(tmp_path, labels_kind="deepcell")

        assert raised.value.filename == str(artifact)

    @pytest.mark.parametrize(
        "wrong", ["cells.h5ad", "ome", "masks_deepcell", "ome/sample-a.ome.tiff", "masks_deepcell/sample-a.tiff"]
    )
    def test_wrong_artifact_kind(self, tmp_path: Path, wrong: str) -> None:
        _write_paths(tmp_path)
        artifact = tmp_path / wrong
        directory = artifact.is_dir()
        if directory:
            for child in artifact.iterdir():
                child.unlink()
            artifact.rmdir()
            artifact.touch()
        else:
            artifact.unlink()
            artifact.mkdir()

        with pytest.raises(NotADirectoryError if directory else IsADirectoryError):
            _discover_layout(tmp_path, labels_kind="deepcell")

    @pytest.mark.parametrize("missing", ["image", "mask", "all"])
    def test_missing_pairs_fail(self, tmp_path: Path, missing: str) -> None:
        _write_paths(tmp_path)
        if missing in {"image", "all"}:
            (tmp_path / "ome" / "sample-a.ome.tiff").unlink()
        if missing in {"mask", "all"}:
            (tmp_path / "masks_deepcell" / "sample-b.tiff").unlink()
        if missing == "all":
            (tmp_path / "ome" / "sample-b.ome.tiff").unlink()
            (tmp_path / "masks_deepcell" / "sample-a.tiff").unlink()

        with pytest.raises(ReaderFormatError, match="at least one sample"):
            _discover_layout(tmp_path, labels_kind="deepcell")

    def test_empty_sample_name(self, tmp_path: Path) -> None:
        _write_paths(tmp_path)
        (tmp_path / "ome" / ".ome.tiff").touch()
        (tmp_path / "masks_deepcell" / ".tiff").touch()

        with pytest.raises(ReaderFormatError, match="nonempty name"):
            _discover_layout(tmp_path, labels_kind="deepcell")

    @pytest.mark.parametrize("relative", ["cells.h5ad", "ome/sample-a.ome.tiff", "masks_deepcell/sample-a.tiff"])
    def test_artifact_symlink_escape(self, tmp_path: Path, relative: str) -> None:
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        _write_paths(dataset)
        outside = tmp_path / "outside"
        outside.touch()
        artifact = dataset / relative
        artifact.unlink()
        artifact.symlink_to(outside)

        with pytest.raises(ReaderFormatError, match="outside"):
            _discover_layout(dataset, labels_kind="deepcell")

    @pytest.mark.parametrize("directory", ["ome", "masks_deepcell"])
    def test_directory_symlink_escape(self, tmp_path: Path, directory: str) -> None:
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        _write_paths(dataset)
        outside = tmp_path / "outside"
        artifact = dataset / directory
        artifact.rename(outside)
        artifact.symlink_to(outside, target_is_directory=True)

        with pytest.raises(ReaderFormatError, match="outside"):
            _discover_layout(dataset, labels_kind="deepcell")

    def test_root_not_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "not-a-directory"
        root.touch()

        with pytest.raises(NotADirectoryError):
            _discover_layout(root, labels_kind="deepcell")
