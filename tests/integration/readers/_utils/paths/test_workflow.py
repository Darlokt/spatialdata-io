"""Composition tests for reader-owned layout discovery."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.paths import (
    normalize_dataset_root,
    require_directory,
    require_file,
    require_unique_path,
)

if TYPE_CHECKING:
    import scanpy as sc

    from spatialdata_io.readers._utils.image import inspect_tiff, read_jpeg, read_png
    from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile, DatasetRoot

    def _consumer_type_contract(
        file_path: ArtifactFile,
        directory: ArtifactDirectory,
        root: DatasetRoot,
    ) -> None:
        """Prove that validated paths compose with current ecosystem boundaries."""
        _ = sc.read_10x_h5(file_path)
        _ = sc.read_10x_mtx(directory)
        _ = sc.read_10x_mtx(root)
        _ = inspect_tiff(file_path)
        _ = read_png(file_path)
        _ = read_jpeg(file_path)


class TestPathWorkflow:
    def test_discovers_and_validates_one_file(self, tmp_path: Path) -> None:
        context = ReaderErrorContext(reader="example", artifact="counts")
        (tmp_path / "notes.txt").touch()
        counts = tmp_path / "counts.csv"
        counts.touch()
        root = normalize_dataset_root(tmp_path, context=context)

        selected = require_unique_path(
            (path for path in Path(root).iterdir() if path.suffix == ".csv"),
            context=context,
            search_path=Path(root),
        )

        assert require_file(selected, context=context) == counts.resolve()

    def test_discovers_and_validates_one_directory(self, tmp_path: Path) -> None:
        context = ReaderErrorContext(reader="example", artifact="analysis outputs")
        analysis = tmp_path / "analysis-v2"
        analysis.mkdir()
        (tmp_path / "analysis-notes.txt").touch()
        root = normalize_dataset_root(tmp_path, context=context)

        selected = require_unique_path(
            (path for path in Path(root).iterdir() if path.name.startswith("analysis-") and path.suffix == ""),
            context=context,
            search_path=Path(root),
        )

        assert require_directory(selected, context=context) == analysis.resolve()

    def test_canonical_result_is_stable_after_alias_retarget(self, tmp_path: Path) -> None:
        context = ReaderErrorContext(reader="example", artifact="transcripts")
        internal = tmp_path / "internal.csv"
        internal.write_text("internal")
        external = tmp_path / "external.csv"
        external.write_text("external")
        alias = tmp_path / "transcripts.csv"
        alias.symlink_to(internal)

        validated = require_file(alias, context=context)
        alias.unlink()
        alias.symlink_to(external)

        assert validated.read_text() == "internal"
