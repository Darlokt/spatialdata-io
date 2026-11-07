"""Discover the CODEX cell table, stitched image and optional experiment file of one region.

A processor-style region directory holds one regional cell table and at most
one stitched image; other files are ignored::

    region/
    ├── reg001_compensated.csv     # cell table: exactly one *.csv or *.fcs
    ├── stitched.tif               # optional: zero or one *.tif, *.tiff or *.btf
    └── experiment.json            # never discovered; only read when selected

Suffixes are matched case-insensitively. Several tables (for example
compensated and uncompensated, or CSV and FCS) or several images are ambiguous
and require ``cell_table_file`` or ``image_file``; which one to use is a
scientific choice, so none is preferred. The experiment file is only read when
the caller selects it, because its association with the table cannot be
inferred from the directory. Relative file options resolve against the dataset
directory; absolute ones may point anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactFile,
    list_directory,
    normalize_dataset_root,
    optional_unique_path,
    require_caller_file,
    require_metadata_file,
    require_unique_path,
)

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.paths import PathPolicy

_CELL_SUFFIXES = frozenset({".csv", ".fcs"})
_IMAGE_SUFFIXES = frozenset({".tif", ".tiff", ".btf"})
_LAYOUT_CONTEXT = ReaderErrorContext("codex", "dataset layout")
_CELL_CONTEXT = ReaderErrorContext("codex", "cell table")
_IMAGE_CONTEXT = ReaderErrorContext("codex", "stitched image")
_EXPERIMENT_CONTEXT = ReaderErrorContext("codex", "experiment calibration")


@dataclass(frozen=True, slots=True)
class CodexLayout:
    """Validated source files of one region, without open resources or loaded data."""

    cell_table: ArtifactFile
    image: ArtifactFile | None
    experiment: ArtifactFile | None


def discover_layout(
    path: str | Path,
    *,
    cell_table_file: str | Path | None,
    image_file: str | Path | None,
    experiment_file: str | Path | None = None,
    path_policy: PathPolicy = "strict",
) -> CodexLayout:
    """Resolve exactly one cell table, zero or one stitched image and the selected experiment."""
    root = normalize_dataset_root(path, context=_LAYOUT_CONTEXT, path_policy=path_policy)
    candidates = list_directory(root.path, context=_LAYOUT_CONTEXT)

    if cell_table_file is None:
        candidate = require_unique_path(
            (item for item in candidates if item.suffix.lower() in _CELL_SUFFIXES),
            context=_CELL_CONTEXT,
            search_path=root.path,
        )
        cell_table = require_metadata_file(root, candidate, context=_CELL_CONTEXT)
    else:
        cell_table = require_caller_file(root, cell_table_file, context=_CELL_CONTEXT)
    if cell_table.suffix.lower() not in _CELL_SUFFIXES:
        raise ReaderFormatError(
            context=_CELL_CONTEXT, path=cell_table, found=cell_table.suffix, expected="a .csv or .fcs cell table"
        )

    if image_file is None:
        image_candidate = optional_unique_path(
            (item for item in candidates if item.suffix.lower() in _IMAGE_SUFFIXES),
            context=_IMAGE_CONTEXT,
            search_path=root.path,
        )
        image = (
            None if image_candidate is None else require_metadata_file(root, image_candidate, context=_IMAGE_CONTEXT)
        )
    else:
        image = require_caller_file(root, image_file, context=_IMAGE_CONTEXT)
    if image is not None and image.suffix.lower() not in _IMAGE_SUFFIXES:
        raise ReaderFormatError(
            context=_IMAGE_CONTEXT, path=image, found=image.suffix, expected="a TIFF, OME-TIFF, BigTIFF, or BTF image"
        )

    experiment = (
        None if experiment_file is None else require_caller_file(root, experiment_file, context=_EXPERIMENT_CONTEXT)
    )
    return CodexLayout(cell_table, image, experiment)
