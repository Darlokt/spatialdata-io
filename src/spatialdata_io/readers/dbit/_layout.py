"""Discover the DBiT expression table, barcode map and optional tissue image.

DBiT exports carry no pipeline version, so there is one unversioned layout.
Only the top level of the dataset directory is searched::

    dataset/
    ├── sample.h5ad                  # required: the only *.h5ad (suffix case-insensitive)
    ├── barcode_list.txt             # required: the only name containing "barcode_list"
    └── tissue_lowres_image.png      # optional: the only name ending in "tissue_lowres_image.png"

Several matches for an artifact are ambiguous and fail; the caller then names
the artifact with ``table_file``, ``barcode_positions_file`` or ``image_file``.
An absent discovered image is simply omitted, but an explicitly named image must
exist. A named image may be a TIFF, OME-TIFF, BigTIFF/BTF, PNG or JPEG file.

Discovered artifacts follow the dataset root's ``path_policy``. Relative
overrides resolve against the dataset root and may not leave it; absolute
overrides may point anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactFile,
    normalize_dataset_root,
    optional_unique_path,
    require_caller_file,
    require_metadata_file,
    require_unique_path,
    sorted_paths,
)

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.paths import PathPolicy

_IMAGE_SUFFIXES = frozenset({".tif", ".tiff", ".btf", ".png", ".jpg", ".jpeg"})
_LAYOUT_CONTEXT = ReaderErrorContext("dbit", "dataset layout")
_TABLE_CONTEXT = ReaderErrorContext("dbit", "expression table")
_BARCODE_CONTEXT = ReaderErrorContext("dbit", "barcode positions")
_IMAGE_CONTEXT = ReaderErrorContext("dbit", "histology image")


@dataclass(frozen=True, slots=True)
class DbitLayout:
    """Validated artifact paths, without loaded data or open resources."""

    table: ArtifactFile
    barcode_positions: ArtifactFile
    image: ArtifactFile | None


def discover_layout(
    path: str | Path,
    *,
    table_file: str | Path | None,
    barcode_positions_file: str | Path | None,
    image_file: str | Path | None,
    path_policy: PathPolicy = "strict",
) -> DbitLayout:
    """Resolve one expression table, one barcode map and zero or one image."""
    root = normalize_dataset_root(path, context=_LAYOUT_CONTEXT, path_policy=path_policy)
    candidates = sorted_paths(root.path.iterdir(), context=_LAYOUT_CONTEXT, search_path=root.path)

    if table_file is None:
        table_candidate = require_unique_path(
            (item for item in candidates if item.suffix.lower() == ".h5ad"),
            context=_TABLE_CONTEXT,
            search_path=root.path,
        )
        table = require_metadata_file(root, table_candidate, context=_TABLE_CONTEXT)
    else:
        table = require_caller_file(root, table_file, context=_TABLE_CONTEXT)
    if table.suffix.lower() != ".h5ad":
        raise ReaderFormatError(
            context=_TABLE_CONTEXT, path=table, found=table.suffix, expected="an .h5ad expression table"
        )

    if barcode_positions_file is None:
        barcode_candidate = require_unique_path(
            (item for item in candidates if "barcode_list" in item.name),
            context=_BARCODE_CONTEXT,
            search_path=root.path,
        )
        barcode_positions = require_metadata_file(root, barcode_candidate, context=_BARCODE_CONTEXT)
    else:
        barcode_positions = require_caller_file(root, barcode_positions_file, context=_BARCODE_CONTEXT)

    if image_file is None:
        image_candidate = optional_unique_path(
            (item for item in candidates if item.name.endswith("tissue_lowres_image.png")),
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
            context=_IMAGE_CONTEXT,
            path=image,
            found=image.suffix,
            expected="a TIFF-family, PNG, or JPEG histology image",
        )

    return DbitLayout(table, barcode_positions, image)
