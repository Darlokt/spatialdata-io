"""Discover the requested products of one Space Ranger HD output directory once.

Space Ranger 3.0 and 3.1 write binned outputs::

    outs/
    ├── hd_feature_slice.h5                    # or feature_slice.h5
    ├── binned_outputs/
    │   ├── square_002um/                      # native grid; always required
    │   │   ├── filtered_feature_bc_matrix.h5
    │   │   ├── raw_feature_bc_matrix.h5
    │   │   └── spatial/
    │   │       ├── tissue_positions.parquet
    │   │       ├── scalefactors_json.json
    │   │       ├── tissue_hires_image.png     # optional
    │   │       ├── tissue_lowres_image.png    # optional
    │   │       └── cytassist_image.tiff       # optional
    │   ├── square_008um/                      # same structure
    │   └── square_016um/                      # same; any stored even size such as square_006um
    ├── spatial/                               # optional root copies of the three images
    ├── tissue_image.tif                       # optional full-resolution input (or .btf, .tiff)
    └── microscope_image/                      # optional alternative full-resolution location

Space Ranger 4.0 and 4.1 add segmentation::

    outs/
    ├── feature_slice.h5
    ├── barcode_mappings.parquet               # or segmented_outputs/barcode_mappings.parquet
    ├── binned_outputs/                        # as above
    └── segmented_outputs/
        ├── filtered_feature_cell_matrix.h5
        ├── raw_feature_cell_matrix.h5
        ├── cell_segmentations.geojson
        └── nucleus_segmentations.geojson

10x sample downloads may prefix the root files (``<name>_feature_slice.h5``,
``<name>_barcode_mappings.parquet``, ``<name>_tissue_image.tif``). Discovery
lists the root, ``binned_outputs/`` and ``microscope_image/`` once each and never
searches recursively. Only the selected products are resolved:

- the 2 µm positions and scale factors always (they calibrate every read);
- each selected bin size's ``filtered``/``raw`` matrix, positions and scale
  factors;
- cell boundaries and the selected cell matrix, nucleus boundaries, and for
  nucleus counts the raw 2 µm matrix plus the barcode mappings;
- the feature slice when CytAssist is requested or no count matrix is
  selected, else when present;
- images: each of hires, lowres and CytAssist from root ``spatial/`` when
  present there, else from the 2 µm ``spatial/``; the full-resolution image
  from ``fullres_image_file`` or the one candidate among root
  ``[<name>_]tissue_image.{btf,tif,tiff}`` and the supported images directly in
  ``microscope_image/``.

Several candidates for one artifact raise. Discovery then reads the identity
of every consumed count matrix and feature slice once (``_metadata``), so the
detected Space Ranger version, if any, names every later error.
``fullres_image_file`` resolves a relative path against the dataset root and
accepts an absolute path anywhere; discovered artifacts follow the root's path
policy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactFile,
    DatasetRoot,
    list_directory,
    normalize_dataset_root,
    optional_metadata_directory,
    optional_metadata_file,
    optional_unique_path,
    require_caller_file,
    require_metadata_directory,
    require_metadata_file,
)
from spatialdata_io.readers.visium_hd._metadata import reconcile_identities

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from spatialdata_io.readers._utils.paths import ArtifactDirectory, PathPolicy
    from spatialdata_io.readers.visium_hd._metadata import Identity

_FULLRES_NAME = re.compile(r"(?:.+_)?tissue_image\.(?:btf|tif|tiff)", flags=re.IGNORECASE)
_FULLRES_SUFFIXES = frozenset({".btf", ".tif", ".tiff", ".png", ".jpg", ".jpeg"})
_BIN_DIRECTORY = re.compile(r"square_(\d{3,})um")


@dataclass(frozen=True, slots=True)
class BinLayout:
    """One selected bin size and count population."""

    size_um: int
    counts: ArtifactFile
    positions: ArtifactFile
    scale_factors: ArtifactFile


@dataclass(frozen=True, slots=True)
class SegmentationLayout:
    """Independent requested segmentation products."""

    counts: ArtifactFile | None
    cell_boundaries: ArtifactFile | None
    nucleus_boundaries: ArtifactFile | None


@dataclass(frozen=True, slots=True)
class ImageLayout:
    """Selected optional rasters."""

    hires: ArtifactFile | None
    lowres: ArtifactFile | None
    cytassist: ArtifactFile | None
    fullres: ArtifactFile | None


@dataclass(frozen=True, slots=True)
class NucleusCountSource:
    """The two required artifacts for aggregation of nucleus counts."""

    raw_two_micron_counts: ArtifactFile
    barcode_mappings: ArtifactFile


@dataclass(frozen=True, slots=True)
class VisiumHDLayout:
    """Validated immutable paths and the reconciled source identity."""

    identity: Identity
    feature_slice: ArtifactFile | None
    base_positions: ArtifactFile
    base_scale_factors: ArtifactFile
    bins: tuple[BinLayout, ...]
    segmentation: SegmentationLayout | None
    nucleus_counts: NucleusCountSource | None
    images: ImageLayout | None

    @property
    def version(self) -> str | None:
        """Return the normalized Space Ranger version the consumed sources agree on, if any."""
        return self.identity.format_version


def _context(artifact: str) -> ReaderErrorContext:
    return ReaderErrorContext(reader="visium_hd", artifact=artifact)


def _loose_file(
    root: DatasetRoot, names: tuple[str, ...], entries: tuple[Path, ...], *, artifact: str, required: bool
) -> ArtifactFile | None:
    """Resolve one root file under any of ``names``, optionally prefixed by a dataset name."""
    context = _context(artifact)
    candidates = [
        entry for entry in entries if any(entry.name == name or entry.name.endswith(f"_{name}") for name in names)
    ]
    candidate = optional_unique_path(candidates, context=context, search_path=root.path)
    if candidate is None:
        return require_metadata_file(root, names[0], context=context) if required else None
    return require_metadata_file(root, candidate, context=context)


def _boundary(
    root: DatasetRoot, directory: ArtifactDirectory | None, name: str, *, artifact: str, requested: bool | None
) -> ArtifactFile | None:
    """Resolve requested (``True``), optional (``None``) or skipped (``False``) boundaries."""
    if requested is False or directory is None:
        return None

    context = _context(artifact)
    if requested:
        return require_metadata_file(root, directory / name, context=context)
    return optional_metadata_file(root, directory / name, context=context)


def _image_file(  # noqa: PLR0913, RUF100 - each image prefers its own root entry.
    root: DatasetRoot,
    spatial: ArtifactDirectory | None,
    base_spatial: Path,
    name: str,
    *,
    artifact: str,
    required: bool = False,
) -> ArtifactFile | None:
    """Prefer each root image independently, rejecting malformed entries."""
    context = _context(artifact)
    if spatial is not None:
        image = optional_metadata_file(root, spatial / name, context=context)
        if image is not None:
            return image
    path = base_spatial / name
    if required:
        return require_metadata_file(root, path, context=context)
    return optional_metadata_file(root, path, context=context)


def _fullres(root: DatasetRoot, override: str | Path | None, entries: tuple[Path, ...]) -> ArtifactFile | None:
    context = _context("full-resolution image")
    if override is not None:
        return require_caller_file(root, override, context=context)

    candidates = [entry for entry in entries if _FULLRES_NAME.fullmatch(entry.name)]
    folder = optional_metadata_directory(root, "microscope_image", context=context)
    if folder is not None:
        candidates.extend(
            entry for entry in list_directory(folder, context=context) if entry.suffix.lower() in _FULLRES_SUFFIXES
        )

    path = optional_unique_path(candidates, context=context, search_path=root.path)
    return None if path is None else require_metadata_file(root, path, context=context)


def _bin_directories(root: DatasetRoot) -> dict[int, ArtifactDirectory]:
    """Discover only directly nested vendor bin directories."""
    context = _context("bin directories")
    binned = optional_metadata_directory(root, "binned_outputs", context=context)
    directories: dict[int, ArtifactDirectory] = {}
    if binned is None:
        return directories

    for entry in list_directory(binned, context=context):
        match = _BIN_DIRECTORY.fullmatch(entry.name)
        if match is None:
            continue

        size = int(match[1])
        if size <= 0 or size % 2 or size in directories:
            raise ReaderFormatError(
                context=context,
                found=entry.name,
                expected="unique positive even bin sizes",
                path=binned,
            )
        directories[size] = require_metadata_directory(root, entry, context=context)

    return directories


def _nucleus_counts(
    root: DatasetRoot,
    base: ArtifactDirectory,
    segmented: ArtifactDirectory | None,
    entries: tuple[Path, ...],
) -> NucleusCountSource:
    """Resolve the raw 2 µm matrix and the one root or segmented barcode mapping."""
    context = _context("barcode mappings")
    loose = _loose_file(root, ("barcode_mappings.parquet",), entries, artifact="barcode mappings", required=False)
    nested = (
        None
        if segmented is None
        else optional_metadata_file(root, segmented / "barcode_mappings.parquet", context=context)
    )
    if loose is not None and nested is not None:
        raise ReaderFormatError(
            context=context,
            found="root and segmented mappings",
            expected="exactly one barcode_mappings.parquet",
            path=root.path,
        )

    mappings = loose or nested or require_metadata_file(root, "barcode_mappings.parquet", context=context)
    raw = require_metadata_file(root, base / "raw_feature_bc_matrix.h5", context=_context("raw 2 micrometre counts"))
    return NucleusCountSource(raw, mappings)


def discover_layout(  # noqa: PLR0913, RUF100 - Resolve supported product selections once.
    path: str | Path,
    *,
    bin_sizes: Sequence[int] | Literal["all"] = "all",
    bin_counts_kind: Literal["filtered", "raw"] = "filtered",
    include_cell_boundaries: bool | None = None,
    include_nucleus_boundaries: bool | None = None,
    include_nucleus_table: bool = False,
    cell_counts_kind: Literal["filtered", "raw"] = "filtered",
    include_images: bool = True,
    include_cytassist_image: bool = False,
    fullres_image_file: str | Path | None = None,
    path_policy: PathPolicy = "strict",
) -> VisiumHDLayout:
    """Resolve the selected artifacts and reconcile the identity of every consumed source."""
    root = normalize_dataset_root(path, context=_context("dataset root"), path_policy=path_policy)
    entries = list_directory(root.path, context=_context("dataset root"))
    directories = _bin_directories(root)
    selected = tuple(sorted(directories)) if bin_sizes == "all" else tuple(bin_sizes)
    absent = set(selected).difference(directories)
    if absent:
        raise ReaderFormatError(
            context=_context("bin selection"),
            found=f"missing sizes {sorted(absent)}",
            expected=f"available sizes {sorted(directories)}",
            path=root.path,
        )

    # Every product shares the physical frame calibrated from the native 2 µm grid.
    base = directories.get(2) or require_metadata_directory(
        root, "binned_outputs/square_002um", context=_context("2 micrometre bins")
    )
    base_positions = require_metadata_file(
        root, base / "spatial/tissue_positions.parquet", context=_context("tissue positions")
    )
    base_scale = require_metadata_file(root, base / "spatial/scalefactors_json.json", context=_context("scale factors"))

    bins = tuple(
        BinLayout(
            size,
            require_metadata_file(
                root, directories[size] / f"{bin_counts_kind}_feature_bc_matrix.h5", context=_context("bin counts")
            ),
            require_metadata_file(
                root, directories[size] / "spatial/tissue_positions.parquet", context=_context("tissue positions")
            ),
            require_metadata_file(
                root, directories[size] / "spatial/scalefactors_json.json", context=_context("scale factors")
            ),
        )
        for size in selected
    )

    nucleus_requested = True if include_nucleus_table else include_nucleus_boundaries
    segmented_context = _context("segmented outputs")
    segmented = (
        require_metadata_directory(root, "segmented_outputs", context=segmented_context)
        if include_cell_boundaries or nucleus_requested
        else optional_metadata_directory(root, "segmented_outputs", context=segmented_context)
    )
    cell = _boundary(
        root, segmented, "cell_segmentations.geojson", artifact="cell boundaries", requested=include_cell_boundaries
    )
    nucleus = _boundary(
        root, segmented, "nucleus_segmentations.geojson", artifact="nucleus boundaries", requested=nucleus_requested
    )

    cell_counts = None
    if segmented is not None and (cell is not None or include_nucleus_table):
        cell_counts = require_metadata_file(
            root, segmented / f"{cell_counts_kind}_feature_cell_matrix.h5", context=_context("cell counts")
        )
    segmentation = SegmentationLayout(cell_counts, cell, nucleus) if cell is not None or nucleus is not None else None
    nucleus_counts = _nucleus_counts(root, base, segmented, entries) if include_nucleus_table else None

    feature = _loose_file(
        root,
        ("feature_slice.h5", "hd_feature_slice.h5"),
        entries,
        artifact="feature slice",
        required=include_cytassist_image or (not bins and cell_counts is None),
    )

    images = None
    if include_images or include_cytassist_image:
        spatial = optional_metadata_directory(root, "spatial", context=_context("image directory"))
        images = ImageLayout(
            _image_file(root, spatial, base / "spatial", "tissue_hires_image.png", artifact="hires image")
            if include_images
            else None,
            _image_file(root, spatial, base / "spatial", "tissue_lowres_image.png", artifact="lowres image")
            if include_images
            else None,
            _image_file(
                root, spatial, base / "spatial", "cytassist_image.tiff", artifact="CytAssist image", required=True
            )
            if include_cytassist_image
            else None,
            _fullres(root, fullres_image_file, entries) if include_images else None,
        )

    # Only consumed sources establish identity; unselected matrices are never fallbacks.
    sources = [source.counts for source in bins]
    if cell_counts is not None:
        sources.append(cell_counts)
    if nucleus_counts is not None:
        sources.append(nucleus_counts.raw_two_micron_counts)
    if feature is not None:
        sources.append(feature)

    return VisiumHDLayout(
        reconcile_identities(sources), feature, base_positions, base_scale, bins, segmentation, nucleus_counts, images
    )
