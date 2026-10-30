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
    optional_directory,
    optional_file,
    optional_unique_path,
    require_caller_file,
    require_directory,
    require_file,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


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
    """Selected optional rasters and their calibration source."""

    scale_factors: ArtifactFile
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

    root: DatasetRoot
    feature_slice: ArtifactFile | None
    identity_file: ArtifactFile
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
    candidate = optional_unique_path(candidates, context=_context(names[0]), search_path=root)
    if candidate is None:
        return require_file(root / names[0], context=_context(names[0])) if required else None
    return require_file(candidate, context=_context(names[0]))


def _boundary(directory: Path | None, name: str, *, requested: bool | None) -> ArtifactFile | None:
    if requested is False:
        return None
    if directory is None:
        if requested:
            msg = f"visium_hd: requested {name}, but segmented_outputs is absent"
            raise FileNotFoundError(msg)
        return None

    path = directory / name
    return require_file(path, context=_context(name)) if requested else optional_file(path, context=_context(name))


def _fullres(root: DatasetRoot, override: str | Path | None, entries: tuple[Path, ...]) -> ArtifactFile | None:
    if override is not None:
        return require_caller_file(root, override, context=_context("full-resolution image"))

    candidates: list[Path] = [
        entry
        for entry in entries
        if re.fullmatch(r"(?:.+_)?tissue_image\.(?:btf|tif|tiff)", entry.name, flags=re.IGNORECASE)
    ]

    folder = optional_directory(root / "microscope_image", context=_context("microscope_image"))
    if folder is not None:
        candidates.extend(
            entry
            for entry in folder.iterdir()
            if entry.suffix.lower() in {".btf", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        )

    path = optional_unique_path(candidates, context=_context("full-resolution image"), search_path=root)
    return None if path is None else require_file(path, context=_context("full-resolution image"))


def _bin_directories(root: DatasetRoot) -> dict[int, Path]:
    """Discover only directly nested vendor bin directories."""
    binned = optional_directory(root / "binned_outputs", context=_context("binned outputs"))
    directories: dict[int, Path] = {}
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

                directories[size] = require_directory(entry, context=_context("bin directory"))

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
) -> VisiumHDLayout:
    """Resolve selected standard and loose-download artifacts without recursion."""
    root = normalize_dataset_root(path, context=_context("dataset root"))
    entries: tuple[Path, ...] = tuple(root.iterdir())
    directories = _bin_directories(root)
    selected = tuple(sorted(directories)) if bin_sizes == "all" else tuple(bin_sizes)
    absent = set(selected).difference(directories)
    if absent:
        raise ReaderFormatError(
            context=_context("bin selection"),
            found=f"missing sizes {sorted(absent)}",
            expected=f"available sizes {sorted(directories)}",
            path=root,
        )

    # Every product shares the physical frame calibrated from the native 2 micrometre grid.
    base = directories.get(2)
    if base is None:
        msg = f"visium_hd: missing square_002um calibration artifacts in {root}"
        raise FileNotFoundError(msg)

    base_positions = require_file(base / "spatial/tissue_positions.parquet", context=_context("base positions"))
    base_scale = require_file(base / "spatial/scalefactors_json.json", context=_context("base scale factors"))

    bins = tuple(
        BinLayout(
            size,
            require_file(directories[size] / f"{bin_counts_kind}_feature_bc_matrix.h5", context=_context("bin counts")),
            require_file(directories[size] / "spatial/tissue_positions.parquet", context=_context("bin positions")),
            require_file(directories[size] / "spatial/scalefactors_json.json", context=_context("bin scale factors")),
        )
        for size in selected
    )

    segmented = optional_directory(root / "segmented_outputs", context=_context("segmented outputs"))
    cell = _boundary(segmented, "cell_segmentations.geojson", requested=include_cell_boundaries)
    nucleus = _boundary(
        segmented,
        "nucleus_segmentations.geojson",
        requested=True if include_nucleus_table else include_nucleus_boundaries,
    )

    cell_counts = None
    if cell is not None or include_nucleus_table:
        if segmented is None:
            msg = "visium_hd: missing requested segmented_outputs"
            raise FileNotFoundError(msg)
        cell_counts = require_file(
            segmented / f"{cell_counts_kind}_feature_cell_matrix.h5", context=_context("cell counts")
        )
    segmentation = SegmentationLayout(cell_counts, cell, nucleus) if cell is not None or nucleus is not None else None

    nucleus_counts = None
    if include_nucleus_table:
        loose = _loose_file(root, ("barcode_mappings.parquet",), entries, required=False)
        nested = (
            optional_file(segmented / "barcode_mappings.parquet", context=_context("barcode mappings"))
            if segmented is not None
            else None
        )
        if loose is not None and nested is not None:
            raise ReaderFormatError(
                context=_context("barcode mappings"),
                found="root and segmented mappings",
                expected="exactly one barcode_mappings.parquet",
                path=root,
            )
        mappings = loose if loose is not None else nested
        if mappings is None:
            mappings = require_file(root / "barcode_mappings.parquet", context=_context("barcode mappings"))

        nucleus_counts = NucleusCountSource(
            require_file(base / "raw_feature_bc_matrix.h5", context=_context("raw two-micron counts")), mappings
        )

    identity = bins[0].counts if bins else cell_counts
    if identity is None:
        identity = require_file(base / "filtered_feature_bc_matrix.h5", context=_context("count identity"))
    feature = _loose_file(root, ("feature_slice.h5", "hd_feature_slice.h5"), entries, required=include_cytassist_image)

    images = None
    if include_images or include_cytassist_image:
        spatial = optional_directory(root / "spatial", context=_context("image directory"))
        image_dir = spatial if spatial is not None else base / "spatial"
        images = ImageLayout(
            base_scale,
            optional_file(image_dir / "tissue_hires_image.png", context=_context("hires image"))
            if include_images
            else None,
            optional_file(image_dir / "tissue_lowres_image.png", context=_context("lowres image"))
            if include_images
            else None,
            require_file(image_dir / "cytassist_image.tiff", context=_context("CytAssist image"))
            if include_cytassist_image
            else None,
            _fullres(root, fullres_image_file, entries) if include_images else None,
        )

    return VisiumHDLayout(
        root, feature, identity, base_positions, base_scale, bins, segmentation, nucleus_counts, images
    )
