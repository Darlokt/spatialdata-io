"""CODEX layout, source schemas, identifier precision, and image options."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.codex._layout import _discover_layout
from tests.integration.readers.codex._fixtures import write_csv, write_image

if TYPE_CHECKING:
    from pathlib import Path


class TestDiscoverLayout:
    """Reject missing, ambiguous, wrong-kind, and escaping artifacts."""

    def test_immutable_canonical_layout_and_ignored_backups(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "cells.CSV")
        (tmp_path / "cells.csv.bak").touch()
        layout = _discover_layout(tmp_path, cell_table_file=None, image_file=None)
        assert layout.root == tmp_path.resolve()
        assert layout.cell_table == (tmp_path / "cells.CSV").resolve()
        assert layout.image is None
        assert not hasattr(layout, "__dict__")

    def test_missing_table(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="cell table"):
            _discover_layout(tmp_path, cell_table_file=None, image_file=None)

    def test_wrong_kind(self, tmp_path: Path) -> None:
        (tmp_path / "cells.csv").mkdir()
        with pytest.raises(IsADirectoryError):
            _discover_layout(tmp_path, cell_table_file=None, image_file=None)

    def test_ambiguous_image_and_relative_override(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "cells.csv")
        write_image(tmp_path / "a.tif")
        write_image(tmp_path / "z.tif")
        with pytest.raises(ReaderFormatError, match="2 candidates"):
            _discover_layout(tmp_path, cell_table_file=None, image_file=None)
        layout = _discover_layout(tmp_path, cell_table_file="cells.csv", image_file="z.tif")
        assert layout.image == (tmp_path / "z.tif").resolve()

    @pytest.mark.parametrize("artifact", ["cells.csv", "image.tif"])
    def test_discovered_symlink_must_stay_in_root(self, tmp_path: Path, artifact: str) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        if artifact != "cells.csv":
            write_csv(root / "cells.csv")
        (tmp_path / artifact).touch()
        (root / artifact).symlink_to(tmp_path / artifact)
        with pytest.raises(ReaderFormatError, match="outside"):
            _discover_layout(root, cell_table_file=None, image_file=None)

    @pytest.mark.parametrize("artifact", ["table", "image"])
    def test_override_suffix_validation(self, tmp_path: Path, artifact: str) -> None:
        write_csv(tmp_path / "cells.csv")
        (tmp_path / "bad.txt").touch()
        with pytest.raises(ReaderFormatError, match=r"bad\.txt"):
            _discover_layout(
                tmp_path,
                cell_table_file="bad.txt" if artifact == "table" else None,
                image_file="bad.txt" if artifact == "image" else None,
            )
