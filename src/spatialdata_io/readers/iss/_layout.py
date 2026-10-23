"""Resolve the three caller-selected ISS artifacts once."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.paths import (
    ArtifactFile,
    DatasetRoot,
    normalize_dataset_root,
    require_caller_file,
)

if TYPE_CHECKING:
    from pathlib import Path

_TIFF_SUFFIXES = frozenset({".tif", ".tiff", ".btf"})


@dataclass(frozen=True, slots=True)
class _IssLayout:
    """Own validated canonical paths without retaining open resources."""

    root: DatasetRoot
    raw_image: ArtifactFile
    labels: ArtifactFile
    table: ArtifactFile


def _discover_layout(
    path: str | Path, *, raw_image_file: str | Path, labels_file: str | Path, table_file: str | Path
) -> _IssLayout:
    """Resolve root-relative or absolute caller files into one immutable layout."""
    root = normalize_dataset_root(path, context=ReaderErrorContext("iss", "dataset layout"))
    raw_image = require_caller_file(root, raw_image_file, context=ReaderErrorContext("iss", "raw image"))
    labels = require_caller_file(root, labels_file, context=ReaderErrorContext("iss", "labels"))
    table = require_caller_file(root, table_file, context=ReaderErrorContext("iss", "counts"))

    for artifact in (raw_image, labels):
        if artifact.suffix.lower() not in _TIFF_SUFFIXES:
            message = f"ISS raster {artifact} must be a TIFF, OME-TIFF, BigTIFF, or BTF file."
            raise ValueError(message)
    if table.suffix.lower() != ".h5ad":
        message = f"ISS table {table} must be an h5ad file."
        raise ValueError(message)

    return _IssLayout(root, raw_image, labels, table)
