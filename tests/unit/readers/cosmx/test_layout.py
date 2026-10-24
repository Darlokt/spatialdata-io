"""Deterministic CosMx artifact discovery, prefixes, compression, and FOV rasters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.cosmx._layout import discover_layout

if TYPE_CHECKING:
    from pathlib import Path


def _touch(root: Path, *relative: str) -> Path:
    for name in relative:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return root


_CORE = ("Run_1_exprMat_file.csv", "Run_1_metadata_file.csv")


class TestDiscoverLayout:
    """One inventory per call; optional artifacts are absent rather than guessed."""

    def test_complete_export_in_numeric_fov_order(self, tmp_path: Path) -> None:
        root = _touch(
            tmp_path,
            *_CORE,
            "Run_1_fov_positions_file.csv.gz",
            "Run_1_tx_file.csv",
            "Run_1-polygons.csv.gz",
            "CellComposite/CellComposite_F010.jpg",
            "CellComposite/CellComposite_F002.PNG",
            "CellComposite/.DS_Store",
            "CellLabels/CellLabels_F002.tif",
            "CellLabels/CellLabels_F010.TIFF",
            "CellOverlay/CellOverlay_F002.jpg",
        )
        layout = discover_layout(root, dataset_id=None, include_transcripts=True)
        assert layout.dataset_id == "Run_1"
        assert layout.counts.name == "Run_1_exprMat_file.csv"
        assert layout.fov_positions is not None
        assert layout.fov_positions.name == "Run_1_fov_positions_file.csv.gz"
        assert layout.transcripts is not None
        assert layout.transcripts.name == "Run_1_tx_file.csv"
        assert [(item.fov, item.path.name) for item in layout.images] == [
            (2, "CellComposite_F002.PNG"),
            (10, "CellComposite_F010.jpg"),
        ]
        assert [item.fov for item in layout.labels] == [2, 10]

    def test_minimal_export_without_transcripts(self, tmp_path: Path) -> None:
        layout = discover_layout(_touch(tmp_path, *_CORE), dataset_id=None, include_transcripts=False)
        assert layout.fov_positions is None
        assert layout.transcripts is None
        assert layout.images == ()
        assert layout.labels == ()

    def test_explicit_prefix_selects_one_of_several_exports(self, tmp_path: Path) -> None:
        root = _touch(tmp_path, *_CORE, "Run_2_exprMat_file.csv.gz", "Run_2_metadata_file.csv.gz")
        layout = discover_layout(root, dataset_id="Run_2", include_transcripts=False)
        assert layout.counts.name == "Run_2_exprMat_file.csv.gz"

    @pytest.mark.parametrize(
        ("files", "match"),
        [
            ((), "0 count matrices"),
            ((*_CORE, "Run_2_exprMat_file.csv"), "2 count matrices"),
            ((*_CORE, "Run_1_metadata_file.csv.gz"), "one plain or compressed"),
            ((*_CORE, "CellLabels/CellLabels_F002.jpg"), "CellLabels"),
            ((*_CORE, "CellComposite/composite.jpg"), "CellComposite"),
            ((*_CORE, "CellComposite/CellComposite_F000.jpg"), "CellComposite"),
            ((*_CORE, "CellComposite/A_F2.jpg", "CellComposite/B_F002.png"), "two files for FOV 2"),
        ],
    )
    def test_rejects(self, tmp_path: Path, files: tuple[str, ...], match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            discover_layout(_touch(tmp_path, *files), dataset_id=None, include_transcripts=False)

    @pytest.mark.parametrize("missing", ["Run_1_metadata_file.csv", "Run_1_tx_file.csv"])
    def test_required_files(self, tmp_path: Path, missing: str) -> None:
        root = _touch(tmp_path, *(name for name in _CORE if name != missing))
        with pytest.raises(FileNotFoundError, match=missing):
            discover_layout(root, dataset_id=None, include_transcripts=True)

    @pytest.mark.parametrize(("dataset_id", "error"), [("", ValueError), ("a/b", ValueError), (3, TypeError)])
    def test_invalid_prefix(self, tmp_path: Path, dataset_id: object, error: type[Exception]) -> None:
        with pytest.raises(error):
            discover_layout(tmp_path, dataset_id=dataset_id, include_transcripts=False)  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.
