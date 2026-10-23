"""Discover deterministic instrument and VPT MERSCOPE product artifacts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactFile,
    DatasetRoot,
    normalize_dataset_root,
    require_caller_directory,
    require_metadata_directory,
    require_metadata_file,
    require_unique_path,
)

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.paths import ArtifactDirectory

_MOSAIC_NAME = re.compile(r"mosaic_(?P<stain>.+)_z(?P<z>[0-9]+)\.tif")


@dataclass(frozen=True, slots=True)
class Selection:
    """Own inclusion choices and the requested image-plane order."""

    images: bool
    boundaries: bool
    table: bool
    transcripts: bool
    z_layers: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class Names:
    """Retain source or caller names independently of derived element keys."""

    region: str
    slide: str

    @property
    def dataset_id(self) -> str:
        """Return the common element-key prefix."""
        return f"{self.slide}_{self.region}"


@dataclass(frozen=True, slots=True)
class TableFiles:
    """Keep counts and their corresponding cell annotations together."""

    counts: ArtifactFile
    metadata: ArtifactFile


@dataclass(frozen=True, slots=True)
class TranscriptSource:
    """Identify the event file, bounded vocabulary and assignment population."""

    path: ArtifactFile
    vocabulary: ArtifactFile
    assignment: Literal["current", "original"]


@dataclass(frozen=True, slots=True)
class BoundarySource:
    """Identify an instrument or explicitly selected VPT micron-space geometry."""

    kind: Literal["instrument", "cellpose", "watershed"]
    path: ArtifactFile


@dataclass(frozen=True, slots=True)
class PhysicalSources:
    """Require calibration alongside requested micron-coordinate products."""

    transform: ArtifactFile
    boundaries: BoundarySource | None
    transcripts: TranscriptSource | None


@dataclass(frozen=True, slots=True)
class MosaicChannel:
    """Pair a scientific stain name with one validated single-plane TIFF."""

    stain: str
    path: ArtifactFile


@dataclass(frozen=True, slots=True)
class MosaicLayer:
    """Keep lexical stain order within one explicitly indexed image plane."""

    z: int
    channels: tuple[MosaicChannel, ...]


@dataclass(frozen=True, slots=True)
class InstrumentLayout:
    """Describe requested products from the original instrument region."""

    root: DatasetRoot
    names: Names
    tables: TableFiles | None
    images: tuple[MosaicLayer, ...]
    physical: PhysicalSources | None
    kind: Literal["instrument"] = field(default="instrument", init=False)


@dataclass(frozen=True, slots=True)
class VptLayout:
    """Keep the instrument region and caller-selected replacement analysis distinct."""

    root: DatasetRoot
    vpt_root: DatasetRoot
    names: Names
    tables: TableFiles | None
    images: tuple[MosaicLayer, ...]
    physical: PhysicalSources | None
    kind: Literal["vpt"] = field(default="vpt", init=False)


type MerscopeLayout = InstrumentLayout | VptLayout


def make_selection(
    *,
    include_images: bool,
    include_cell_boundaries: bool,
    include_table: bool,
    include_transcripts: bool,
    z_layers: Sequence[int] | None,
) -> Selection:
    """Validate public inclusion choices and own ordered image-plane indexes."""
    for name, value in (
        ("include_images", include_images),
        ("include_cell_boundaries", include_cell_boundaries),
        ("include_table", include_table),
        ("include_transcripts", include_transcripts),
    ):
        if not isinstance(value, bool):
            message = f"{name} must be a boolean."
            raise TypeError(message)
    selected = None
    if z_layers is not None:
        if isinstance(z_layers, str | bytes) or not isinstance(z_layers, Sequence):
            message = "z_layers must be an ordered sequence of nonnegative integer indexes or None."
            raise TypeError(message)
        selected = tuple(z_layers)
        if not selected or any(isinstance(z, bool) or not isinstance(z, int) or z < 0 for z in selected):
            message = "z_layers must contain nonnegative integer indexes and cannot be empty."
            raise ValueError(message)
        if len(set(selected)) != len(selected):
            message = "z_layers cannot contain duplicate indexes."
            raise ValueError(message)
    return Selection(include_images, include_cell_boundaries, include_table, include_transcripts, selected)


def _validate_name(name: str | None, parameter: str) -> None:
    """Reject unusable name overrides before filesystem discovery."""
    if name is None:
        return
    if not isinstance(name, str):
        message = f"{parameter} must be a string or None."
        raise TypeError(message)
    if not name.strip() or name in {".", ".."} or any(character in name for character in "/\\\x00"):
        message = f"{parameter} must be nonempty and cannot contain path separators."
        raise ValueError(message)


def _file(root: DatasetRoot, name: str) -> ArtifactFile:
    """Resolve vendor artifact names inside their owning region or analysis."""
    return require_metadata_file(root, name, context=ReaderErrorContext("merscope", name))


def _vpt_boundaries(root: DatasetRoot, entries: frozenset[str]) -> BoundarySource:
    """Reject absent or ambiguous supported VPT boundary outputs."""
    candidates = [
        BoundarySource(kind, _file(root, f"{kind}_micron_space.parquet"))
        for kind in ("cellpose", "watershed")
        if f"{kind}_micron_space.parquet" in entries
    ]
    require_unique_path(
        (source.path for source in candidates),
        context=ReaderErrorContext("merscope", "VPT micron-space boundaries"),
        search_path=root,
    )
    return candidates[0]


def _mosaic_layers(
    root: DatasetRoot, directory: ArtifactDirectory, z_layers: tuple[int, ...] | None
) -> tuple[MosaicLayer, ...]:
    """Discover stain/plane pairs once and reject missing or aliased selections."""
    context = ReaderErrorContext("merscope", "mosaic image grid")
    files: dict[tuple[str, int], ArtifactFile] = {}
    for entry in sorted(directory.iterdir()):
        match = _MOSAIC_NAME.fullmatch(entry.name)
        if match is None:
            continue
        key = (match["stain"], int(match["z"]))
        if key in files:
            raise ReaderFormatError(
                context=context,
                path=directory,
                found=f"multiple files for stain/z {key!r}",
                expected="one file per stain/z pair",
            )
        files[key] = _file(root, f"images/{entry.name}")
    if not files:
        raise ReaderFormatError(
            context=context,
            path=directory,
            found="no supported mosaic files",
            expected="mosaic_<stain>_z<index>.tif files",
        )
    stains = sorted({stain for stain, _ in files})
    selected = tuple(sorted({z for _, z in files})) if z_layers is None else z_layers
    missing = [(stain, z) for z in selected for stain in stains if (stain, z) not in files]
    if missing:
        raise ReaderFormatError(
            context=context,
            path=directory,
            found=f"missing stain/z pairs {missing!r}",
            expected="every requested plane for every stain",
        )
    return tuple(MosaicLayer(z, tuple(MosaicChannel(stain, files[stain, z]) for stain in stains)) for z in selected)


def discover_layout(
    path: str | Path,
    selection: Selection,
    *,
    vpt_directory: str | Path | None = None,
    region_name: str | None = None,
    slide_name: str | None = None,
) -> MerscopeLayout:
    """Resolve requested instrument/VPT products without interpreting pixels or rows."""
    _validate_name(region_name, "region_name")
    _validate_name(slide_name, "slide_name")
    root = normalize_dataset_root(path, context=ReaderErrorContext("merscope", "region root"))
    names = Names(
        root.name if region_name is None else region_name, root.parent.name if slide_name is None else slide_name
    )

    vpt_root = None
    if vpt_directory is not None:
        context = ReaderErrorContext("merscope", "VPT directory")
        directory = require_caller_directory(root, vpt_directory, context=context)
        vpt_root = normalize_dataset_root(directory, context=context)

    analysis = root if vpt_root is None else vpt_root
    tables = (
        TableFiles(_file(analysis, "cell_by_gene.csv"), _file(analysis, "cell_metadata.csv"))
        if selection.table
        else None
    )

    entries = (
        frozenset(entry.name for entry in analysis.iterdir())
        if vpt_root is not None and (selection.boundaries or selection.transcripts)
        else frozenset()
    )

    boundaries = None
    if selection.boundaries:
        boundaries = (
            BoundarySource("instrument", _file(root, "cell_boundaries.parquet"))
            if vpt_root is None
            else _vpt_boundaries(analysis, entries)
        )

    transcripts = None
    if selection.transcripts:
        current = vpt_root is None or "detected_transcripts.csv" in entries
        source_root = analysis if current else root
        transcripts = TranscriptSource(
            _file(source_root, "detected_transcripts.csv"),
            _file(source_root, "cell_by_gene.csv"),
            "current" if current else "original",
        )

    images: tuple[MosaicLayer, ...] = ()
    physical = None
    if selection.images or boundaries is not None or transcripts is not None:
        image_directory = require_metadata_directory(root, "images", context=ReaderErrorContext("merscope", "images"))
        if selection.images:
            images = _mosaic_layers(root, image_directory, selection.z_layers)
        if boundaries is not None or transcripts is not None:
            physical = PhysicalSources(
                _file(root, "images/micron_to_mosaic_pixel_transform.csv"), boundaries, transcripts
            )

    if vpt_root is None:
        return InstrumentLayout(root, names, tables, images, physical)
    return VptLayout(root, vpt_root, names, tables, images, physical)
