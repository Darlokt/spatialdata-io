"""Discover the requested products in a Space Ranger HD output tree once."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactFile,
    DatasetRoot,
    normalize_dataset_root,
    optional_metadata_directory,
    optional_metadata_file,
    optional_unique_path,
    require_caller_file,
    require_metadata_directory,
    require_metadata_file,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from spatialdata_io.readers._utils.paths import ArtifactDirectory, PathPolicy


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
    """Validated immutable paths, without live files or parsed metadata."""

    feature_slice: ArtifactFile | None
    base_positions: ArtifactFile
    base_scale_factors: ArtifactFile
    bins: tuple[BinLayout, ...]
    segmentation: SegmentationLayout | None
    nucleus_counts: NucleusCountSource | None
    images: ImageLayout | None


def _context(artifact: str) -> ReaderErrorContext:
    return ReaderErrorContext(reader="visium_hd", artifact=artifact)


def _loose_file(
    root: DatasetRoot, names: tuple[str, ...], entries: tuple[Path, ...], *, required: bool
) -> ArtifactFile | None:

    candidates: list[Path] = [
        entry for entry in entries if any(entry.name == name or entry.name.endswith(f"_{name}") for name in names)
    ]
    candidate = optional_unique_path(candidates, context=_context(names[0]), search_path=root.path)
    if candidate is None:
        return require_metadata_file(root, names[0], context=_context(names[0])) if required else None
    return require_metadata_file(root, candidate, context=_context(names[0]))


def _boundary(
    root: DatasetRoot, directory: ArtifactDirectory | None, name: str, *, requested: bool | None
) -> ArtifactFile | None:
    if requested is False:
        return None
    if directory is None:
        if requested:
            msg = f"visium_hd: requested {name}, but segmented_outputs is absent"
            raise FileNotFoundError(msg)
        return None

    path = directory / name
    if requested:
        return require_metadata_file(root, path, context=_context(name))
    return optional_metadata_file(root, path, context=_context(name))


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
    if override is not None:
        return require_caller_file(root, override, context=_context("full-resolution image"))

    candidates: list[Path] = [
        entry
        for entry in entries
        if re.fullmatch(r"(?:.+_)?tissue_image\.(?:btf|tif|tiff)", entry.name, flags=re.IGNORECASE)
    ]

    folder = optional_metadata_directory(root, "microscope_image", context=_context("microscope_image"))
    if folder is not None:
        candidates.extend(
            entry
            for entry in folder.iterdir()
            if entry.suffix.lower() in {".btf", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        )

    path = optional_unique_path(candidates, context=_context("full-resolution image"), search_path=root.path)
    return None if path is None else require_metadata_file(root, path, context=_context("full-resolution image"))


def _bin_directories(root: DatasetRoot) -> dict[int, ArtifactDirectory]:
    """Discover only directly nested vendor bin directories."""
    binned = optional_metadata_directory(root, "binned_outputs", context=_context("binned outputs"))
    directories: dict[int, ArtifactDirectory] = {}
    if binned is not None:
        for entry in sorted(binned.iterdir()):
            match = re.fullmatch(r"square_(\d{3,})um", entry.name)
            if match is not None:
                size = int(match[1])
                if size <= 0 or size % 2 or size in directories:
                    raise ReaderFormatError(
                        context=_context("bin directories"),
                        found=entry.name,
                        expected="unique positive even bin sizes",
                        path=binned,
                    )

                directories[size] = require_metadata_directory(root, entry, context=_context("bin directory"))

    return directories


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
    """Resolve selected standard and loose-download artifacts without recursion."""
    root = normalize_dataset_root(path, context=_context("dataset root"), path_policy=path_policy)
    entries: tuple[Path, ...] = tuple(sorted(root.path.iterdir()))
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

    # Every product shares the physical frame calibrated from the native 2 micrometre grid.
    base = directories.get(2)
    if base is None:
        msg = f"visium_hd: missing square_002um calibration artifacts in {root.path}"
        raise FileNotFoundError(msg)

    base_positions = require_metadata_file(
        root, base / "spatial/tissue_positions.parquet", context=_context("base positions")
    )
    base_scale = require_metadata_file(
        root, base / "spatial/scalefactors_json.json", context=_context("base scale factors")
    )

    bins = tuple(
        BinLayout(
            size,
            require_metadata_file(
                root, directories[size] / f"{bin_counts_kind}_feature_bc_matrix.h5", context=_context("bin counts")
            ),
            require_metadata_file(
                root, directories[size] / "spatial/tissue_positions.parquet", context=_context("bin positions")
            ),
            require_metadata_file(
                root, directories[size] / "spatial/scalefactors_json.json", context=_context("bin scale factors")
            ),
        )
        for size in selected
    )

    segmented = optional_metadata_directory(root, "segmented_outputs", context=_context("segmented outputs"))
    cell = _boundary(root, segmented, "cell_segmentations.geojson", requested=include_cell_boundaries)
    nucleus = _boundary(
        root,
        segmented,
        "nucleus_segmentations.geojson",
        requested=True if include_nucleus_table else include_nucleus_boundaries,
    )

    cell_counts = None
    if cell is not None or include_nucleus_table:
        if segmented is None:
            msg = "visium_hd: missing requested segmented_outputs"
            raise FileNotFoundError(msg)
        cell_counts = require_metadata_file(
            root, segmented / f"{cell_counts_kind}_feature_cell_matrix.h5", context=_context("cell counts")
        )
    segmentation = SegmentationLayout(cell_counts, cell, nucleus) if cell is not None or nucleus is not None else None

    nucleus_counts = None
    if include_nucleus_table:
        loose = _loose_file(root, ("barcode_mappings.parquet",), entries, required=False)
        nested = (
            optional_metadata_file(root, segmented / "barcode_mappings.parquet", context=_context("barcode mappings"))
            if segmented is not None
            else None
        )
        if loose is not None and nested is not None:
            raise ReaderFormatError(
                context=_context("barcode mappings"),
                found="root and segmented mappings",
                expected="exactly one barcode_mappings.parquet",
                path=root.path,
            )
        mappings = loose if loose is not None else nested
        if mappings is None:
            mappings = require_metadata_file(root, "barcode_mappings.parquet", context=_context("barcode mappings"))

        nucleus_counts = NucleusCountSource(
            require_metadata_file(root, base / "raw_feature_bc_matrix.h5", context=_context("raw two-micron counts")),
            mappings,
        )

    feature = _loose_file(
        root,
        ("feature_slice.h5", "hd_feature_slice.h5"),
        entries,
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

    return VisiumHDLayout(feature, base_positions, base_scale, bins, segmentation, nucleus_counts, images)
