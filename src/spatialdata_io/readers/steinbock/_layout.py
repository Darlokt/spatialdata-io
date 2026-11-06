"""Discover one Steinbock dataset: the cell table, paired OME images and masks, ``images.csv``.

Steinbock processes multiplexed (typically imaging mass cytometry, IMC) images
in one project directory. Its documented file types are
https://bodenmillergroup.github.io/steinbock/latest/file-types/. One image is
identified throughout by its TIFF file name ``<image>.tiff``: in
``images.csv``, in the per-image CSV files, in the table's ``obs["image"]`` and
in its observation names ``Object <id> in <image>.tiff``. Of a complete
project, the reader uses only the exported OME images, one segmentation-mask
directory, the exported AnnData table and ``images.csv``::

    project/
    ├── panel.csv                 # channel panel; not read (OME carries the names)
    ├── images.csv                # optional; dimensions and IMC acquisition extents
    ├── img/<image>.tiff          # channel-first images; not read (see ome/)
    ├── masks/<image>.tiff        # required: segmentation masks (masks_dir)
    ├── intensities/<image>.csv   # per-object intensities; not read (see the table)
    ├── regionprops/<image>.csv   # per-object properties; not read (see the table)
    ├── neighbors/<image>.csv     # object neighbour pairs; not read
    ├── ome/<image>.ome.tiff      # required: ``steinbock export ome`` of img/
    └── cells.h5ad                # required: ``steinbock export anndata`` (table_file)

``masks/`` is the default output of ``steinbock segment``. A project that keeps
several segmentations side by side (for example ``masks_deepcell/`` and
``masks_ilastik/``) selects one with ``masks_dir``, a directory path inside the
project that follows the dataset's path policy like every discovered artifact.
``steinbock export anndata`` has no default file name, so ``table_file`` names
the table: relative to the project (under the path policy) or absolute.

Pairing rules:

- A sample is the file name without its exact suffix: ``<sample>.ome.tiff`` in
  ``ome/`` and ``<sample>.tiff`` in the mask directory. Other files, including
  ``.tif`` spellings and backups such as ``<sample>.tiff.bak``, are ignored.
- Every image needs exactly one mask and every mask exactly one image; an
  unmatched file fails instead of being omitted, and at least one pair is
  required. The table must describe exactly these samples (checked when it is
  read).
- Sample names become element names (``<sample>_image``, ``<sample>_labels``)
  and frame names, so they must be valid SpatialData names (letters, digits,
  ``_``, ``-`` and ``.``; not ``.``, ``..`` or a leading ``__``) and distinct
  ignoring case.
- A sample's frames are ``<sample>`` (micrometres, when several samples are
  read), ``<sample>_pixels`` (raster pixels of an uncalibrated sample) and
  ``<sample>_regionprops_indices`` (table centroids). Sample names allow ``_``,
  so no suffix can be collision-free by construction: a sample named like
  another sample's derived frame (``a_pixels`` next to ``a``) is rejected,
  whatever the calibration, so one frame name never means two units.
- Samples are ordered by the lexical order of their image file names.

There is no version marker, so no format version is detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.names import validate_element_name
from spatialdata_io.readers._utils.paths import (
    ArtifactFile,
    normalize_dataset_root,
    optional_metadata_file,
    require_caller_file,
    require_metadata_directory,
    require_metadata_file,
    sorted_paths,
)
from spatialdata_io.readers.steinbock._schemas import CELL_TABLE_CONTEXT, IMAGE_METADATA_CONTEXT

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import PathPolicy

PIXEL_FRAME_SUFFIX = "_pixels"
NATIVE_FRAME_SUFFIX = "_regionprops_indices"

_IMAGE_SUFFIX = ".ome.tiff"
_LABEL_SUFFIX = ".tiff"
_LAYOUT_CONTEXT = ReaderErrorContext("steinbock", "dataset layout")
_IMAGES_CONTEXT = ReaderErrorContext("steinbock", "OME images")
_MASKS_CONTEXT = ReaderErrorContext("steinbock", "masks")


@dataclass(frozen=True, slots=True)
class SampleLayout:
    """One sample name with its validated OME image and mask."""

    name: str
    image: ArtifactFile
    labels: ArtifactFile


@dataclass(frozen=True, slots=True)
class SteinbockLayout:
    """The cell table, the samples in image file-name order and the optional ``images.csv``."""

    table: ArtifactFile
    samples: tuple[SampleLayout, ...]
    image_metadata: ArtifactFile | None


def validate_masks_dir(masks_dir: object) -> str:
    """Return the public ``masks_dir`` option: a relative directory path that stays inside the project."""
    if not isinstance(masks_dir, str):
        message = f"masks_dir must be a string, not {type(masks_dir).__name__}."
        raise TypeError(message)
    parts = PurePath(masks_dir).parts
    if not parts or PurePath(masks_dir).anchor or ".." in parts:
        message = f"masks_dir must be a relative directory inside the dataset without '..', not {masks_dir!r}."
        raise ValueError(message)
    return masks_dir


def validate_table_file(table_file: object) -> Path:
    """Return the public ``table_file`` option as a path after checking its type."""
    if not isinstance(table_file, str | Path):
        message = f"table_file must be a str or Path, not {type(table_file).__name__}."
        raise TypeError(message)
    return Path(table_file)


def discover_layout(
    path: str | Path,
    *,
    masks_dir: str = "masks",
    table_file: str | Path = "cells.h5ad",
    path_policy: PathPolicy = "strict",
) -> SteinbockLayout:
    """Resolve the table and a nonempty exact pairing of images and masks."""
    root = normalize_dataset_root(path, context=_LAYOUT_CONTEXT, path_policy=path_policy)
    table = require_caller_file(root, table_file, context=CELL_TABLE_CONTEXT)
    images_directory = require_metadata_directory(root, "ome", context=_IMAGES_CONTEXT)
    masks_directory = require_metadata_directory(root, masks_dir, context=_MASKS_CONTEXT)

    images = {
        candidate.name.removesuffix(_IMAGE_SUFFIX): require_metadata_file(root, candidate, context=_IMAGES_CONTEXT)
        for candidate in sorted_paths(images_directory.iterdir(), context=_IMAGES_CONTEXT, search_path=images_directory)
        if candidate.name.endswith(_IMAGE_SUFFIX)
    }
    masks = {
        candidate.name.removesuffix(_LABEL_SUFFIX): require_metadata_file(root, candidate, context=_MASKS_CONTEXT)
        for candidate in sorted_paths(masks_directory.iterdir(), context=_MASKS_CONTEXT, search_path=masks_directory)
        if candidate.name.endswith(_LABEL_SUFFIX)
    }
    if not images or "" in images or "" in masks or images.keys() != masks.keys():
        raise ReaderFormatError(
            context=_LAYOUT_CONTEXT,
            path=root.path,
            found=(
                f"{len(images)} images and {len(masks)} masks. "
                f"Unmasked images {sorted(images.keys() - masks.keys())!r}, "
                f"unmatched masks {sorted(masks.keys() - images.keys())!r}"
            ),
            expected="at least one sample with a nonempty name and exactly one image and mask",
        )

    _validate_sample_names(list(images), root.path)

    return SteinbockLayout(
        table,
        tuple(SampleLayout(name, image, masks[name]) for name, image in images.items()),
        optional_metadata_file(root, "images.csv", context=IMAGE_METADATA_CONTEXT),
    )


def _validate_sample_names(names: list[str], root: Path) -> None:
    """Require valid, case-insensitively distinct names that no derived frame name repeats."""
    for name in names:
        try:
            validate_element_name(name, parameter="sample name")
        except ValueError as error:
            raise ReaderFormatError(
                context=_LAYOUT_CONTEXT,
                path=root,
                found=f"invalid sample name {name!r}",
                expected="sample names that are valid SpatialData element and coordinate-system names",
            ) from error

    folded = {name.lower() for name in names}
    if len(folded) != len(names):
        raise ReaderFormatError(
            context=_LAYOUT_CONTEXT,
            path=root,
            found=f"case-insensitively conflicting sample names {names!r}",
            expected="case-insensitively distinct sample names",
        )

    conflicts = sorted(
        frame
        for name in names
        for frame in (f"{name}{PIXEL_FRAME_SUFFIX}", f"{name}{NATIVE_FRAME_SUFFIX}")
        if frame.lower() in folded
    )
    if conflicts:
        raise ReaderFormatError(
            context=_LAYOUT_CONTEXT,
            path=root,
            found=f"derived frame names {conflicts!r} that are also sample names",
            expected=(f"no sample named '<other sample>{PIXEL_FRAME_SUFFIX}' or '<other sample>{NATIVE_FRAME_SUFFIX}'"),
        )
