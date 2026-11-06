"""Composition tests for reader-owned layout discovery."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    normalize_dataset_root,
    require_caller_file,
    require_metadata_directory,
    require_metadata_file,
    require_unique_path,
)

if TYPE_CHECKING:
    import scanpy as sc

    from spatialdata_io.readers._utils.image import inspect_tiff, read_jpeg, read_png
    from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile, DatasetRoot, PathPolicy

    def _consumer_type_contract(
        file_path: ArtifactFile,
        directory: ArtifactDirectory,
        root: DatasetRoot,
    ) -> None:
        """Prove that validated paths compose with current ecosystem boundaries."""
        _ = sc.read_10x_h5(file_path)
        _ = sc.read_10x_mtx(directory)
        _ = sc.read_10x_mtx(root.path)
        _ = inspect_tiff(file_path, context=_RASTER_CONTEXT)
        _ = read_png(file_path, context=_RASTER_CONTEXT)
        _ = read_jpeg(file_path, context=_RASTER_CONTEXT)


_RASTER_CONTEXT = ReaderErrorContext("test", "raster")
_CONTEXT = ReaderErrorContext(reader="example", artifact="counts")


def _annexed_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Build a git-annex style dataset whose files link into an object store outside the root."""
    repository = tmp_path / "repository"
    objects = repository / ".git" / "annex" / "objects" / "Xk" / "4m"
    objects.mkdir(parents=True)
    target = objects / "SHA256E-s9--5e1f.csv"
    target.write_text("a,b\n1,2\n")
    root = repository / "dataset"
    (root / "outs").mkdir(parents=True)
    (root / "outs" / "counts.csv").symlink_to(target)
    (root / "notes.txt").touch()
    return root, target


class TestPathWorkflow:
    def test_discovers_and_validates_one_file(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").touch()
        counts = tmp_path / "counts.csv"
        counts.touch()
        root = normalize_dataset_root(tmp_path, context=_CONTEXT, path_policy="strict")

        selected = require_unique_path(
            (path for path in root.path.iterdir() if path.suffix == ".csv"),
            context=_CONTEXT,
            search_path=root.path,
        )

        assert require_metadata_file(root, selected, context=_CONTEXT) == counts.resolve()

    def test_discovers_and_validates_one_directory(self, tmp_path: Path) -> None:
        context = ReaderErrorContext(reader="example", artifact="analysis outputs")
        analysis = tmp_path / "analysis-v2"
        analysis.mkdir()
        (tmp_path / "analysis-notes.txt").touch()
        root = normalize_dataset_root(tmp_path, context=context, path_policy="strict")

        selected = require_unique_path(
            (path for path in root.path.iterdir() if path.name.startswith("analysis-") and path.suffix == ""),
            context=context,
            search_path=root.path,
        )

        assert require_metadata_directory(root, selected, context=context) == analysis.resolve()

    def test_canonical_result_is_stable_after_alias_retarget(self, tmp_path: Path) -> None:
        context = ReaderErrorContext(reader="example", artifact="transcripts")
        internal = tmp_path / "internal.csv"
        internal.write_text("internal")
        external = tmp_path / "external.csv"
        external.write_text("external")
        alias = tmp_path / "transcripts.csv"
        alias.symlink_to(internal)
        root = normalize_dataset_root(tmp_path, context=context, path_policy="strict")

        validated = require_metadata_file(root, alias, context=context)
        alias.unlink()
        alias.symlink_to(external)

        assert validated.read_text() == "internal"


class TestAnnexedDatasetWorkflow:
    """One annexed dataset under both policies: the root decides, discovery code is unchanged."""

    @staticmethod
    def _discover(path: Path, path_policy: PathPolicy) -> ArtifactFile:
        root = normalize_dataset_root(path, context=_CONTEXT, path_policy=path_policy)
        outs = require_metadata_directory(root, "outs", context=_CONTEXT)
        candidate = require_unique_path(
            (entry for entry in outs.iterdir() if entry.suffix == ".csv"), context=_CONTEXT, search_path=outs
        )
        return require_metadata_file(root, candidate, context=_CONTEXT)

    def test_strict_rejects_the_annexed_artifact(self, tmp_path: Path) -> None:
        root, target = _annexed_dataset(tmp_path)

        with pytest.raises(ReaderFormatError, match="resolving outside the dataset root") as raised:
            self._discover(root, "strict")

        assert str(target) in str(raised.value)
        assert raised.value.path == root.resolve() / "outs" / "counts.csv"

    def test_relaxed_reads_the_annexed_artifact_under_its_dataset_name(self, tmp_path: Path) -> None:
        root, _ = _annexed_dataset(tmp_path)

        counts = self._discover(root, "relaxed")

        assert counts == root.resolve() / "outs" / "counts.csv"
        assert counts.read_text() == "a,b\n1,2\n"

    @pytest.mark.parametrize("path_policy", ["strict", "relaxed"])
    def test_metadata_named_escapes_are_rejected_under_both_policies(
        self, tmp_path: Path, path_policy: PathPolicy
    ) -> None:
        path, target = _annexed_dataset(tmp_path)
        root = normalize_dataset_root(path, context=_CONTEXT, path_policy=path_policy)
        escapes = (Path("..") / ".git" / "annex" / "objects" / "Xk" / "4m" / target.name, str(target))

        for escape in escapes:
            with pytest.raises(ReaderFormatError, match="lexically outside the dataset root"):
                require_metadata_file(root, escape, context=_CONTEXT)

    @pytest.mark.parametrize("path_policy", ["strict", "relaxed"])
    def test_caller_overrides_under_both_policies(self, tmp_path: Path, path_policy: PathPolicy) -> None:
        path, target = _annexed_dataset(tmp_path)
        root = normalize_dataset_root(path, context=_CONTEXT, path_policy=path_policy)

        # Absolute caller paths are unrestricted and keep the name the caller gave.
        assert require_caller_file(root, target, context=_CONTEXT) == target
        assert require_caller_file(root, path / "outs" / "counts.csv", context=_CONTEXT).name == "counts.csv"
        with pytest.raises(ReaderFormatError, match="lexically outside the dataset root"):
            require_caller_file(root, Path("..") / ".git", context=_CONTEXT)
        # Relative caller paths follow the root's policy.
        if path_policy == "strict":
            with pytest.raises(ReaderFormatError, match="resolving outside the dataset root"):
                require_caller_file(root, "outs/counts.csv", context=_CONTEXT)
        else:
            counts = require_caller_file(root, "outs/counts.csv", context=_CONTEXT)
            assert counts == path.resolve() / "outs" / "counts.csv"
            assert counts.read_text() == "a,b\n1,2\n"
