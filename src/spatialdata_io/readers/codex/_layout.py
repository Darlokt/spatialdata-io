"""Deterministic CODEX table and stitched-image discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactFile,
    DatasetRoot,
    normalize_dataset_root,
    optional_unique_path,
    require_caller_file,
    require_metadata_file,
    require_unique_path,
    sorted_paths,
)

if TYPE_CHECKING:
    from pathlib import Path

_CELL_SUFFIXES = frozenset({".csv", ".fcs"})
_IMAGE_SUFFIXES = frozenset({".tif", ".tiff", ".btf"})


@dataclass(frozen=True, slots=True)
class _CodexLayout:
    """Hold canonical validated source paths without open resources or loaded data."""

    root: DatasetRoot
    cell_table: ArtifactFile
    image: ArtifactFile | None
    experiment: ArtifactFile | None


def _discover_layout(
    path: str | Path,
    *,
    cell_table_file: str | Path | None,
    image_file: str | Path | None,
    experiment_file: str | Path | None = None,
) -> _CodexLayout:
    """Resolve exactly one selected cell table and zero or one stitched image."""
    root = normalize_dataset_root(path, context=ReaderErrorContext("codex", "dataset layout"))
    candidates = sorted_paths(root.iterdir(), context=ReaderErrorContext("codex", "dataset layout"), search_path=root)

    cell_context = ReaderErrorContext("codex", "cell table")
    if cell_table_file is None:
        candidate = require_unique_path(
            (item for item in candidates if item.suffix.lower() in _CELL_SUFFIXES),
            context=cell_context,
            search_path=root,
        )
        cell_table = require_metadata_file(root, candidate, context=cell_context)
    else:
        cell_table = require_caller_file(root, cell_table_file, context=cell_context)
    if cell_table.suffix.lower() not in _CELL_SUFFIXES:
        raise ReaderFormatError(
            context=cell_context, path=cell_table, found=cell_table.suffix, expected="a .csv or .fcs cell table"
        )

    image_context = ReaderErrorContext("codex", "stitched image")
    if image_file is None:
        image_candidate = optional_unique_path(
            (item for item in candidates if item.suffix.lower() in _IMAGE_SUFFIXES),
            context=image_context,
            search_path=root,
        )
        image = None if image_candidate is None else require_metadata_file(root, image_candidate, context=image_context)
    else:
        image = require_caller_file(root, image_file, context=image_context)
    if image is not None and image.suffix.lower() not in _IMAGE_SUFFIXES:
        raise ReaderFormatError(
            context=image_context, path=image, found=image.suffix, expected="a TIFF, OME-TIFF, BigTIFF, or BTF image"
        )

    experiment = (
        None
        if experiment_file is None
        else require_caller_file(root, experiment_file, context=ReaderErrorContext("codex", "experiment calibration"))
    )
    return _CodexLayout(root, cell_table, image, experiment)
