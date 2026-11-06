"""Resolve the three caller-selected ISS artifacts into one immutable layout.

ISS pipelines have no fixed directory tree or format version, so nothing is
discovered: the caller names each artifact, relative to the dataset directory or
as an absolute path::

    <path>/
    ├── <raw_image_file>   # TIFF, OME-TIFF, BigTIFF or BTF (.tif, .tiff, .btf)
    ├── <labels_file>      # TIFF-family integer segmentation on the same pixel grid
    └── <table_file>       # AnnData .h5ad: expression and observation metadata

Suffixes are checked on the names the caller gave before any filesystem access.
Relative paths follow the dataset root's ``path_policy``; absolute paths may
point anywhere. The layout keeps paths only, never open files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.paths import ArtifactFile, normalize_dataset_root, require_caller_file

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import PathPolicy

_TIFF_SUFFIXES = frozenset({".tif", ".tiff", ".btf"})


@dataclass(frozen=True, slots=True)
class IssLayout:
    """Validated paths of the raw image, its labels and the linked table."""

    raw_image: ArtifactFile
    labels: ArtifactFile
    table: ArtifactFile


def validate_artifact_names(raw_image_file: str | Path, labels_file: str | Path, table_file: str | Path) -> None:
    """Check the suffixes of the names the caller gave, before any filesystem access."""
    for name, value in (("raw_image_file", raw_image_file), ("labels_file", labels_file)):
        if Path(value).suffix.lower() not in _TIFF_SUFFIXES:
            message = f"ISS {name} {str(value)!r} must be a TIFF, OME-TIFF, BigTIFF or BTF file (.tif, .tiff, .btf)."
            raise ValueError(message)
    if Path(table_file).suffix.lower() != ".h5ad":
        message = f"ISS table_file {str(table_file)!r} must be an .h5ad file."
        raise ValueError(message)


def discover_layout(
    path: str | Path,
    *,
    raw_image_file: str | Path,
    labels_file: str | Path,
    table_file: str | Path,
    path_policy: PathPolicy = "strict",
) -> IssLayout:
    """Resolve the dataset root and each caller file once.

    The file names were checked by :func:`validate_artifact_names`; resolved
    caller files keep the name the caller gave, so their suffixes still hold.
    """
    root = normalize_dataset_root(path, context=ReaderErrorContext("iss", "dataset layout"), path_policy=path_policy)
    return IssLayout(
        require_caller_file(root, raw_image_file, context=ReaderErrorContext("iss", "raw image")),
        require_caller_file(root, labels_file, context=ReaderErrorContext("iss", "labels")),
        require_caller_file(root, table_file, context=ReaderErrorContext("iss", "table")),
    )
