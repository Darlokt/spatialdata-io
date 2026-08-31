"""Tests for the narrow reader-facing path package surface."""

from __future__ import annotations

import ast
from inspect import getattr_static
from pathlib import Path

from spatialdata_io.readers._utils import paths


class TestPathsPackage:
    def test_exports_only_path_contracts(self) -> None:
        assert paths.__all__ == [
            "ArtifactDirectory",
            "ArtifactFile",
            "DatasetRoot",
            "normalize_dataset_root",
            "optional_directory",
            "optional_file",
            "optional_unique_path",
            "require_caller_directory",
            "require_caller_file",
            "require_directory",
            "require_file",
            "require_metadata_directory",
            "require_metadata_file",
            "require_unique_path",
            "sorted_paths",
        ]
        assert not hasattr(paths, "ReaderFormatError")
        assert not hasattr(paths, "resolve_caller_path")

    def test_artifact_types_are_allocation_free_path_markers(self) -> None:
        path = Path("artifact")
        directory = paths.ArtifactDirectory(path)
        file = paths.ArtifactFile(path)
        root = paths.DatasetRoot(directory)

        assert directory is path
        assert file is path
        assert root is path
        assert getattr_static(paths.ArtifactDirectory, "__supertype__") is Path
        assert getattr_static(paths.ArtifactFile, "__supertype__") is Path
        assert getattr_static(paths.DatasetRoot, "__supertype__") is paths.ArtifactDirectory
        assert paths.ArtifactDirectory is not paths.ArtifactFile

    def test_only_path_core_mints_validated_path_markers(self) -> None:
        repository_root = Path(__file__).resolve().parents[5]
        readers = repository_root / "src" / "spatialdata_io" / "readers"
        assert readers.is_dir(), f"reader source directory does not exist: {readers}"
        constructors = {"ArtifactDirectory", "ArtifactFile", "DatasetRoot"}

        for path in readers.rglob("*.py"):
            if path == readers / "_utils" / "paths" / "_core.py":
                continue
            tree = ast.parse(path.read_text())
            calls: set[str] = set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                else:
                    continue
                if name in constructors:
                    calls.add(name)
            assert not calls, f"{path} constructs validated path markers directly: {sorted(calls)}"
