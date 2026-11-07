"""Deterministic CosMx artifact discovery, prefixes, compression, and FOV rasters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers.cosmx._layout import discover_layout, validate_file_prefix

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
        layout = discover_layout(root, file_prefix=None, include_transcripts=True)
        assert layout.file_prefix == "Run_1"
        assert layout.counts.name == "Run_1_exprMat_file.csv"
        assert layout.fov_positions is not None
        assert layout.fov_positions.name == "Run_1_fov_positions_file.csv.gz"
        assert layout.transcripts is not None
        assert layout.transcripts.name == "Run_1_tx_file.csv"
        assert [(item.fov, item.path.name) for item in layout.images] == [
            (2, "CellComposite_F002.PNG"),
            (10, "CellComposite_F010.jpg"),
        ]
        assert layout.labels is not None
        assert layout.labels.directory == root / "CellLabels"
        assert [item.fov for item in layout.labels.files] == [2, 10]

    def test_minimal_export_without_transcripts(self, tmp_path: Path) -> None:
        layout = discover_layout(_touch(tmp_path, *_CORE), file_prefix=None, include_transcripts=False)
        assert layout.fov_positions is None
        assert layout.transcripts is None
        assert layout.images == ()
        assert layout.labels is None

    def test_explicit_prefix_selects_one_of_several_exports(self, tmp_path: Path) -> None:
        root = _touch(tmp_path, *_CORE, "Run_2_exprMat_file.csv.gz", "Run_2_metadata_file.csv.gz")
        layout = discover_layout(root, file_prefix="Run_2", include_transcripts=False)
        assert layout.file_prefix == "Run_2"
        assert layout.counts.name == "Run_2_exprMat_file.csv.gz"

    @pytest.mark.parametrize("prefix", ["Lung 5", "  ", "a:b", "_"])
    def test_any_file_name_prefix_is_discovered_and_selected(self, tmp_path: Path, prefix: str) -> None:
        root = _touch(tmp_path, f"{prefix}_exprMat_file.csv", f"{prefix}_metadata_file.csv")
        for file_prefix in (None, prefix):
            layout = discover_layout(root, file_prefix=file_prefix, include_transcripts=False)
            assert layout.file_prefix == prefix
            assert layout.metadata.name == f"{prefix}_metadata_file.csv"

    @pytest.mark.parametrize("dotfile", [False, True])
    def test_present_empty_labels_fail(self, tmp_path: Path, *, dotfile: bool) -> None:
        root = _touch(tmp_path, *_CORE)
        directory = root / "CellLabels"
        directory.mkdir()
        if dotfile:
            (directory / ".DS_Store").touch()
        with pytest.raises(ReaderFormatError, match="no label rasters") as error:
            discover_layout(root, file_prefix=None, include_transcripts=False)
        assert error.value.path == directory
        assert error.value.context == ReaderErrorContext("cosmx", "cell labels")

    def test_empty_composites_remain_optional(self, tmp_path: Path) -> None:
        root = _touch(tmp_path, *_CORE)
        (root / "CellComposite").mkdir()
        layout = discover_layout(root, file_prefix=None, include_transcripts=False)
        assert layout.images == ()

    def test_partial_labels_are_discovered_without_metadata_parsing(self, tmp_path: Path) -> None:
        root = _touch(tmp_path, *_CORE, "CellLabels/CellLabels_F002.tif")
        layout = discover_layout(root, file_prefix=None, include_transcripts=False)
        assert layout.labels is not None
        assert [item.fov for item in layout.labels.files] == [2]

    @pytest.mark.parametrize("artifact", ["CellLabels", "CellComposite"])
    def test_raster_directory_wrong_kind(self, tmp_path: Path, artifact: str) -> None:
        root = _touch(tmp_path, *_CORE, artifact)
        with pytest.raises(NotADirectoryError) as error:
            discover_layout(root, file_prefix=None, include_transcripts=False)
        assert error.value.filename == str(root / artifact)

    @pytest.mark.parametrize(
        ("files", "artifact", "match"),
        [
            ((), "count matrix", "pass file_prefix otherwise; found 0 count matrices"),
            ((*_CORE, "Run_2_exprMat_file.csv"), "count matrix", "2 count matrices"),
            ((*_CORE, "Run_1_metadata_file.csv.gz"), "cell metadata", "one plain or compressed"),
            ((*_CORE, "CellLabels/CellLabels_F002.jpg"), "cell labels", "CellLabels_F002.jpg"),
            ((*_CORE, "CellComposite/composite.jpg"), "composite images", "composite.jpg"),
            ((*_CORE, "CellComposite/CellComposite_F000.jpg"), "composite images", "positive FOV"),
            ((*_CORE, "CellComposite/A_F2.jpg", "CellComposite/B_F002.png"), "composite images", "two files for FOV 2"),
        ],
    )
    def test_rejects(self, tmp_path: Path, files: tuple[str, ...], artifact: str, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match) as error:
            discover_layout(_touch(tmp_path, *files), file_prefix=None, include_transcripts=False)
        assert error.value.context == ReaderErrorContext("cosmx", artifact)

    def test_explicit_prefix_without_files(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"Run_2_exprMat_file\.csv"):
            discover_layout(_touch(tmp_path, *_CORE), file_prefix="Run_2", include_transcripts=False)

    @pytest.mark.parametrize("missing", ["Run_1_metadata_file.csv", "Run_1_tx_file.csv"])
    def test_required_files(self, tmp_path: Path, missing: str) -> None:
        root = _touch(tmp_path, *(name for name in _CORE if name != missing))
        with pytest.raises(FileNotFoundError, match=missing):
            discover_layout(root, file_prefix=None, include_transcripts=True)


class TestValidateFilePrefix:
    """A caller prefix names files directly inside the export; only its form is checked."""

    @pytest.mark.parametrize("prefix", [None, "Run_1", "Lung 5", " ", "a:b", "..", "_"])
    def test_accepts(self, prefix: str | None) -> None:
        validate_file_prefix(prefix)

    @pytest.mark.parametrize("prefix", ["", "a/b", "a\\b", "a\x00b"])
    def test_rejects_values(self, prefix: str) -> None:
        with pytest.raises(ValueError, match="file_prefix must be nonempty and cannot contain path separators or NUL"):
            validate_file_prefix(prefix)

    @pytest.mark.parametrize("prefix", [3, b"Run_1", ("Run_1",)])
    def test_rejects_types(self, prefix: object) -> None:
        with pytest.raises(TypeError, match=f"file_prefix must be a string or None; found {type(prefix).__name__}"):
            validate_file_prefix(prefix)
