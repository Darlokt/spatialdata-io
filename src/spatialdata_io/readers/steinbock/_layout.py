"""Pair contained Steinbock OME images and selected-family masks deterministically."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactFile,
    DatasetRoot,
    normalize_dataset_root,
    require_metadata_directory,
    require_metadata_file,
    sorted_paths,
)

if TYPE_CHECKING:
    from pathlib import Path

_IMAGE_SUFFIX = ".ome.tiff"
_LABEL_SUFFIX = ".tiff"


@dataclass(frozen=True, slots=True)
class _SampleLayout:
    """Hold one sample name and its validated image and selected mask paths."""

    name: str
    image: ArtifactFile
    labels: ArtifactFile


@dataclass(frozen=True, slots=True)
class _SteinbockLayout:
    """Hold the validated dataset root, owned-table source, and sorted sample pairs."""

    root: DatasetRoot
    table: ArtifactFile
    samples: tuple[_SampleLayout, ...]


def _discover_layout(path: str | Path, *, labels_kind: Literal["deepcell", "ilastik"]) -> _SteinbockLayout:
    """Resolve a nonempty exact image/mask pairing without omitting unmatched samples."""
    if labels_kind not in {"deepcell", "ilastik"}:
        message = "labels_kind must be 'deepcell' or 'ilastik'."
        raise ValueError(message)

    context = ReaderErrorContext("steinbock", "dataset layout")
    root = normalize_dataset_root(path, context=context)
    table = require_metadata_file(root, "cells.h5ad", context=ReaderErrorContext("steinbock", "counts"))
    images_context = ReaderErrorContext("steinbock", "OME images")
    masks_context = ReaderErrorContext("steinbock", f"{labels_kind} masks")
    images_directory = require_metadata_directory(root, "ome", context=images_context)
    masks_directory = require_metadata_directory(root, f"masks_{labels_kind}", context=masks_context)

    # Full suffixes distinguish actual artifacts from unrelated reports and backups.
    images = {
        candidate.name.removesuffix(_IMAGE_SUFFIX): require_metadata_file(root, candidate, context=images_context)
        for candidate in sorted_paths(images_directory.iterdir(), context=images_context, search_path=images_directory)
        if candidate.name.endswith(_IMAGE_SUFFIX)
    }
    masks = {
        candidate.name.removesuffix(_LABEL_SUFFIX): require_metadata_file(root, candidate, context=masks_context)
        for candidate in sorted_paths(masks_directory.iterdir(), context=masks_context, search_path=masks_directory)
        if candidate.name.endswith(_LABEL_SUFFIX)
    }
    if not images or "" in images or "" in masks or images.keys() != masks.keys():
        raise ReaderFormatError(
            context=context,
            path=root,
            found=(
                f"{len(images)} images and {len(masks)} masks. "
                f"Unmasked images {sorted(images.keys() - masks.keys())!r}, "
                f"unmatched masks {sorted(masks.keys() - images.keys())!r}"
            ),
            expected="at least one sample with a nonempty name and exactly one image and selected-family mask",
        )

    # Sample identifiers also name SpatialData elements and coordinate systems.
    invalid_names = [
        name
        for name in images
        if name in {".", ".."}
        or name.startswith("__")
        or any(not character.isalnum() and character not in "_-." for character in name)
    ]
    if invalid_names or len({name.lower() for name in images}) != len(images):
        raise ReaderFormatError(
            context=context,
            path=root,
            found=f"invalid or case-insensitively conflicting sample names {list(images)!r}",
            expected=(
                "case-insensitively distinct sample names containing only alphanumeric characters, '_', '-', or '.', "
                "excluding '.', '..', and names starting with '__'"
            ),
        )

    return _SteinbockLayout(
        root,
        table,
        tuple(_SampleLayout(name, image, masks[name]) for name, image in images.items()),
    )
