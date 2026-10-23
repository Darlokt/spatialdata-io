"""Test deterministic DBiT artifact selection, overrides, and containment."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.dbit._layout import _discover_layout

if TYPE_CHECKING:
    from pathlib import Path


def _write_artifacts(root: Path) -> None:
    (root / "sample.h5ad").touch()
    (root / "barcode_list.txt").touch()


class TestDiscoverLayout:
    """Select artifacts once and reject malformed or ambiguous layouts."""

    def test_immutable_canonical_layout_and_ignored_backups(self, tmp_path: Path) -> None:
        _write_artifacts(tmp_path)
        (tmp_path / "sample.h5ad.bak").touch()
        (tmp_path / "histology.tif").touch()
        (tmp_path / "prefix_tissue_lowres_image.png").touch()

        layout = _discover_layout(tmp_path, table_file=None, barcode_positions_file=None, image_file=None)

        assert layout.root == tmp_path.resolve()
        assert layout.table == (tmp_path / "sample.h5ad").resolve()
        assert layout.barcode_positions == (tmp_path / "barcode_list.txt").resolve()
        assert layout.image == (tmp_path / "prefix_tissue_lowres_image.png").resolve()
        assert not hasattr(layout, "__dict__")

        with pytest.raises(FrozenInstanceError):
            layout.image = None  # type: ignore[misc, read-only]  # Verify the runtime frozen dataclass contract.

    def test_optional_image_absence(self, tmp_path: Path) -> None:
        _write_artifacts(tmp_path)

        layout = _discover_layout(tmp_path, table_file=None, barcode_positions_file=None, image_file=None)

        assert layout.image is None

    @pytest.mark.parametrize("artifact", ["sample.h5ad", "barcode_list.txt"])
    def test_missing_required_artifact(self, tmp_path: Path, artifact: str) -> None:
        _write_artifacts(tmp_path)
        (tmp_path / artifact).unlink()

        with pytest.raises(FileNotFoundError, match="dbit") as caught:
            _discover_layout(tmp_path, table_file=None, barcode_positions_file=None, image_file=None)

        assert caught.value.filename == str(tmp_path)

    @pytest.mark.parametrize("artifact", ["sample.h5ad", "barcode_list.txt", "tissue_lowres_image.png"])
    def test_wrong_kind_artifact(self, tmp_path: Path, artifact: str) -> None:
        _write_artifacts(tmp_path)
        path = tmp_path / artifact
        path.unlink(missing_ok=True)
        path.mkdir()

        with pytest.raises(IsADirectoryError) as caught:
            _discover_layout(tmp_path, table_file=None, barcode_positions_file=None, image_file=None)

        assert caught.value.filename == str(path)

    @pytest.mark.parametrize("artifact", ["sample.h5ad", "barcode_list.txt", "tissue_lowres_image.png"])
    def test_ambiguity_and_relative_override(self, tmp_path: Path, artifact: str) -> None:
        _write_artifacts(tmp_path)
        (tmp_path / "tissue_lowres_image.png").touch()
        (tmp_path / f"other_{artifact}").touch()

        with pytest.raises(ReaderFormatError, match="2 candidates"):
            _discover_layout(tmp_path, table_file=None, barcode_positions_file=None, image_file=None)

        layout = _discover_layout(
            tmp_path,
            table_file="sample.h5ad",
            barcode_positions_file="barcode_list.txt",
            image_file="tissue_lowres_image.png",
        )

        assert layout.table.name == "sample.h5ad"
        assert layout.barcode_positions.name == "barcode_list.txt"
        assert layout.image == tmp_path / "tissue_lowres_image.png"

    def test_absolute_overrides_outside_root(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _write_artifacts(tmp_path)
        (tmp_path / "image.jpeg").touch()

        layout = _discover_layout(
            root,
            table_file=tmp_path / "sample.h5ad",
            barcode_positions_file=tmp_path / "barcode_list.txt",
            image_file=tmp_path / "image.jpeg",
        )

        assert layout.table == tmp_path / "sample.h5ad"
        assert layout.barcode_positions == tmp_path / "barcode_list.txt"
        assert layout.image == tmp_path / "image.jpeg"

    @pytest.mark.parametrize("artifact", ["sample.h5ad", "barcode_list.txt", "tissue_lowres_image.png"])
    def test_discovered_symlink_cannot_escape_root(self, tmp_path: Path, artifact: str) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _write_artifacts(root)
        (tmp_path / artifact).touch()
        (root / artifact).unlink(missing_ok=True)
        (root / artifact).symlink_to(tmp_path / artifact)

        with pytest.raises(ReaderFormatError, match="outside"):
            _discover_layout(root, table_file=None, barcode_positions_file=None, image_file=None)

    def test_relative_override_cannot_escape_root(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _write_artifacts(root)
        (tmp_path / "image.png").touch()

        with pytest.raises(ReaderFormatError, match="outside"):
            _discover_layout(root, table_file=None, barcode_positions_file=None, image_file="../image.png")

    @pytest.mark.parametrize("artifact", ["table", "barcodes", "image"])
    def test_missing_explicit_override_is_not_ignored(self, tmp_path: Path, artifact: str) -> None:
        _write_artifacts(tmp_path)

        with pytest.raises(FileNotFoundError, match="missing"):
            _discover_layout(
                tmp_path,
                table_file="missing.h5ad" if artifact == "table" else None,
                barcode_positions_file="missing.txt" if artifact == "barcodes" else None,
                image_file="missing.png" if artifact == "image" else None,
            )

    @pytest.mark.parametrize("artifact", ["table", "image"])
    def test_override_suffix_validation(self, tmp_path: Path, artifact: str) -> None:
        _write_artifacts(tmp_path)
        (tmp_path / "invalid.txt").touch()

        with pytest.raises(ReaderFormatError, match=r"invalid\.txt"):
            _discover_layout(
                tmp_path,
                table_file="invalid.txt" if artifact == "table" else None,
                barcode_positions_file=None,
                image_file="invalid.txt" if artifact == "image" else None,
            )

    def test_invalid_root(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _discover_layout(tmp_path / "absent", table_file=None, barcode_positions_file=None, image_file=None)

        (tmp_path / "file").touch()
        with pytest.raises(NotADirectoryError):
            _discover_layout(tmp_path / "file", table_file=None, barcode_positions_file=None, image_file=None)
