"""Discover deterministic DBiT tables, barcode positions, and optional histology."""

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

_IMAGE_SUFFIXES = frozenset({".tif", ".tiff", ".btf", ".png", ".jpg", ".jpeg"})


@dataclass(frozen=True, slots=True)
class _DbitLayout:
    """Hold canonical validated artifacts without loaded data or open resources."""

    root: DatasetRoot
    table: ArtifactFile
    barcode_positions: ArtifactFile
    image: ArtifactFile | None


def _discover_layout(
    path: str | Path,
    *,
    table_file: str | Path | None,
    barcode_positions_file: str | Path | None,
    image_file: str | Path | None,
) -> _DbitLayout:
    """Resolve one expression table, one barcode map, and zero or one image."""
    context = ReaderErrorContext("dbit", "dataset layout")
    root = normalize_dataset_root(path, context=context)
    candidates = sorted_paths(root.iterdir(), context=context, search_path=root)

    # Discovery must stay inside the root. Explicit caller overrides use the
    # separate path policy that permits absolute files outside the dataset.
    table_context = ReaderErrorContext("dbit", "expression table")
    if table_file is None:
        table_candidate = require_unique_path(
            (item for item in candidates if item.suffix.lower() == ".h5ad"),
            context=table_context,
            search_path=root,
        )
        table = require_metadata_file(root, table_candidate, context=table_context)
    else:
        table = require_caller_file(root, table_file, context=table_context)
    if table.suffix.lower() != ".h5ad":
        raise ReaderFormatError(
            context=table_context, path=table, found=table.suffix, expected="an .h5ad expression table"
        )

    barcode_context = ReaderErrorContext("dbit", "barcode positions")
    if barcode_positions_file is None:
        barcode_candidate = require_unique_path(
            (item for item in candidates if "barcode_list" in item.name),
            context=barcode_context,
            search_path=root,
        )
        barcode_positions = require_metadata_file(root, barcode_candidate, context=barcode_context)
    else:
        barcode_positions = require_caller_file(root, barcode_positions_file, context=barcode_context)

    # An absent discovered image is optional. Ambiguity or a missing explicit
    # image is an error, so requested content is never silently omitted.
    image_context = ReaderErrorContext("dbit", "histology image")
    if image_file is None:
        image_candidate = optional_unique_path(
            (item for item in candidates if item.name.endswith("tissue_lowres_image.png")),
            context=image_context,
            search_path=root,
        )
        image = None if image_candidate is None else require_metadata_file(root, image_candidate, context=image_context)
    else:
        image = require_caller_file(root, image_file, context=image_context)
    if image is not None and image.suffix.lower() not in _IMAGE_SUFFIXES:
        raise ReaderFormatError(
            context=image_context,
            path=image,
            found=image.suffix,
            expected="a TIFF-family, PNG, or JPEG histology image",
        )

    return _DbitLayout(root, table, barcode_positions, image)
