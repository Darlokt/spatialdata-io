"""Deterministic local-path and contextual-error utilities for readers.

The package owns representation-free filesystem safety. Readers retain all
vendor filenames, matching rules, version semantics, layout interpretation,
and data loading.
"""

from __future__ import annotations

from spatialdata_io.readers._utils.paths._core import (
    DatasetRoot,
    normalize_dataset_root,
    optional_directory,
    optional_file,
    optional_unique_path,
    require_caller_directory,
    require_caller_file,
    require_directory,
    require_file,
    require_metadata_directory,
    require_metadata_file,
    require_unique_path,
    sorted_paths,
)

__all__ = [
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
