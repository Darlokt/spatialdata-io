"""Tests for the narrow reader-facing path package surface."""

from __future__ import annotations

from spatialdata_io.readers._utils import paths


class TestPathsPackage:
    def test_exports_only_path_contracts(self) -> None:
        assert paths.__all__ == [
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
