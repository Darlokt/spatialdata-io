"""Discover one deterministic collection of Spatial Genomics ROI artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactFile,
    DatasetRoot,
    normalize_dataset_root,
    require_metadata_file,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class Selection:
    """Own normalized product and ROI selection without retaining caller lists."""

    images: bool
    labels: bool
    transcripts: bool
    shapes: Literal["none", "polygons", "circles", "both"]
    rois: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class PixelLayout:
    """Require DAPI calibration whenever a pixel-based product is requested."""

    dapi: ArtifactFile
    labels: ArtifactFile | None
    boundaries: ArtifactFile | None


@dataclass(frozen=True, slots=True)
class RoiLayout:
    """Carry validated artifacts for one independently calibrated ROI."""

    name: str
    counts: ArtifactFile
    cells: ArtifactFile
    pixels: PixelLayout | None
    transcripts: ArtifactFile | None


@dataclass(frozen=True, slots=True)
class SeqfishLayout:
    """Own the canonical root and final ordered ROI inventory."""

    root: DatasetRoot
    rois: tuple[RoiLayout, ...]


def make_selection(
    *,
    include_images: bool,
    include_labels: bool,
    include_transcripts: bool,
    include_shapes: bool,
    shape_geometry: Literal["polygons", "circles", "both"],
    rois: Sequence[str] | None,
) -> Selection:
    """Validate public choices before discovery and normalize borrowed sequences."""
    for name, value in (
        ("include_images", include_images),
        ("include_labels", include_labels),
        ("include_transcripts", include_transcripts),
        ("include_shapes", include_shapes),
    ):
        if not isinstance(value, bool):
            message = f"{name} must be a boolean."
            raise TypeError(message)

    if not isinstance(shape_geometry, str):
        message = "shape_geometry must be a string."
        raise TypeError(message)
    if shape_geometry not in {"polygons", "circles", "both"}:
        message = "shape_geometry must be 'polygons', 'circles', or 'both'."
        raise ValueError(message)
    selected = None
    if rois is not None:
        if isinstance(rois, str | bytes) or not isinstance(rois, Sequence):
            message = "rois must be a sequence of ROI names, not a string."
            raise TypeError(message)
        selected = tuple(rois)
        if not selected or any(
            not isinstance(name, str) or not name or name in {".", ".."} or any(c in name for c in "/\\\x00")
            for name in selected
        ):
            message = "rois must contain nonempty filename prefixes without path separators."
            raise ValueError(message)
        if len(set(selected)) != len(selected):
            message = "rois must not contain duplicate ROI names."
            raise ValueError(message)
    return Selection(
        include_images, include_labels, include_transcripts, shape_geometry if include_shapes else "none", selected
    )


def _roi_layout(root: DatasetRoot, name: str, selection: Selection) -> RoiLayout:
    """Resolve exact artifacts; selected products cannot silently disappear."""

    def artifact(suffix: str) -> ArtifactFile:
        """Keep discovered vendor paths inside their normalized dataset root."""
        return require_metadata_file(root, f"{name}_{suffix}", context=ReaderErrorContext("seqfish", suffix))

    counts = artifact("CellxGene.csv")
    cells = artifact("CellCoordinates.csv")
    polygons = selection.shapes in {"polygons", "both"}
    pixels = None
    if selection.images or selection.labels or polygons:
        pixels = PixelLayout(
            artifact("DAPI.tiff"),
            artifact("Segmentation.tiff") if selection.labels else None,
            artifact("Boundaries.geojson") if polygons else None,
        )
    return RoiLayout(name, counts, cells, pixels, artifact("TranscriptList.csv") if selection.transcripts else None)


def discover_layout(path: str | Path, selection: Selection) -> SeqfishLayout:
    """Inspect one directory and resolve the requested complete ROI groups."""
    context = ReaderErrorContext("seqfish", "ROI layout")
    root = normalize_dataset_root(path, context=context)
    names = {
        entry.name.removesuffix(suffix)
        for entry in root.iterdir()
        for suffix in ("_CellCoordinates.csv", "_CellxGene.csv")
        if entry.name.endswith(suffix)
    }
    if not names or "" in names:
        raise ReaderFormatError(
            context=context,
            path=root,
            found="no valid ROI prefixes",
            expected="paired <ROI>_CellxGene/CellCoordinates.csv",
        )
    requested = selection.rois if selection.rois is not None else tuple(sorted(names))
    missing = set(requested).difference(names)
    if missing:
        raise ReaderFormatError(
            context=context, path=root, found=f"unknown ROIs {sorted(missing)!r}", expected=f"ROIs in {sorted(names)!r}"
        )
    return SeqfishLayout(root, tuple(_roi_layout(root, name, selection) for name in requested))
