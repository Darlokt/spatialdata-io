"""Discover the artifacts of one MERSCOPE region and an optional VPT analysis.

Instrument output for one region (``path``)::

    <slide>/
    └── <region>/
        ├── cell_by_gene.csv            # counts; also the transcript vocabulary
        ├── cell_metadata.csv           # cell centroids and annotations
        ├── cell_boundaries.parquet     # one row per cell and z slice
        ├── detected_transcripts.csv    # transcript events
        └── images/
            ├── micron_to_mosaic_pixel_transform.csv   # optional calibration
            └── mosaic_<stain>_z<index>.tif            # one plane per stain and z index

VPT (Vizgen Post-processing Tool) output in ``vpt_directory``, which replaces
the cell products of the region::

    <vpt_directory>/
    ├── cell_by_gene.csv
    ├── cell_metadata.csv
    ├── cellpose_micron_space.parquet   # exactly one of the two segmentations
    ├── watershed_micron_space.parquet
    └── detected_transcripts.csv        # optional, events reassigned to the new cells

Mosaics and their calibration always come from the region. Transcripts come
from the VPT directory when it has its own ``detected_transcripts.csv``
(current assignments); otherwise from the region, whose assignments then refer
to the original instrument cells. Each transcript file is paired with the
``cell_by_gene.csv`` beside it, whose header is the feature vocabulary.

Only the artifacts of selected products are required. File names establish no
instrument or software release, so no format version is detected. Discovered
artifacts follow the region root's ``path_policy``; a relative
``vpt_directory`` resolves against the region root and the VPT directory is
then a second root with the same policy.

Element names start with ``<slide>_<region>``. Without ``slide_name`` and
``region_name`` the parts come from the parent and region directory names,
repaired with a warning when they are not valid element names.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.names import sanitize_element_name, validate_element_name
from spatialdata_io.readers._utils.paths import (
    ArtifactFile,
    DatasetRoot,
    list_directory,
    normalize_dataset_root,
    optional_metadata_file,
    require_caller_directory,
    require_metadata_directory,
    require_metadata_file,
    require_unique_path,
)
from spatialdata_io.readers.merscope._schemas import BOUNDARIES, COUNTS, METADATA, TRANSCRIPTS

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import PathPolicy

_MOSAIC_NAME = re.compile(r"mosaic_(?P<stain>.+)_z(?P<z>[0-9]+)\.tif")
_TRANSFORM_FILE = "images/micron_to_mosaic_pixel_transform.csv"
_SEGMENTATIONS = ("cellpose", "watershed")
_ROOT_CONTEXT = ReaderErrorContext("merscope", "region root")
_VPT_CONTEXT = ReaderErrorContext("merscope", "VPT directory")
_MOSAIC_CONTEXT = ReaderErrorContext("merscope", "mosaic image grid")


@dataclass(frozen=True, slots=True)
class Selection:
    """Requested products and the requested image-plane order (``None`` for all planes)."""

    images: bool
    boundaries: bool
    table: bool
    transcripts: bool
    z_layers: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class Names:
    """Valid slide and region element-name parts."""

    region: str
    slide: str

    @property
    def dataset_id(self) -> str:
        """Return the common element-name prefix ``<slide>_<region>``."""
        return f"{self.slide}_{self.region}"


@dataclass(frozen=True, slots=True)
class TableFiles:
    """Counts and the cell annotations of the same population."""

    counts: ArtifactFile
    metadata: ArtifactFile


@dataclass(frozen=True, slots=True)
class TranscriptSource:
    """Event file, the count file whose header is its vocabulary, and its assignment population."""

    path: ArtifactFile
    vocabulary: ArtifactFile
    assignment: Literal["current", "original"]


@dataclass(frozen=True, slots=True)
class MosaicChannel:
    """One stain's single-plane TIFF."""

    stain: str
    path: ArtifactFile


@dataclass(frozen=True, slots=True)
class MosaicLayer:
    """One z plane with its stains in lexical order."""

    z: int
    channels: tuple[MosaicChannel, ...]


@dataclass(frozen=True, slots=True)
class Mosaics:
    """Selected mosaic planes and the optional micron-to-mosaic calibration beside them."""

    layers: tuple[MosaicLayer, ...]
    transform: ArtifactFile | None


@dataclass(frozen=True, slots=True)
class MerscopeLayout:
    """Artifacts of the selected products from the instrument or a VPT analysis."""

    names: Names
    tables: TableFiles | None
    images: Mosaics | None
    boundaries: ArtifactFile | None
    transcripts: TranscriptSource | None


def _validate_z_layers(z_layers: Sequence[int] | None) -> tuple[int, ...] | None:
    """Return the requested unique nonnegative plane indexes in caller order."""
    if z_layers is None:
        return None
    if isinstance(z_layers, str | bytes) or not isinstance(z_layers, Sequence):
        message = "z_layers must be an ordered sequence of nonnegative integer indexes or None."
        raise TypeError(message)
    planes = tuple(z_layers)
    if any(isinstance(z, bool) or not isinstance(z, int) for z in planes):
        message = "z_layers must contain integer indexes."
        raise TypeError(message)
    if not planes or any(z < 0 for z in planes):
        message = "z_layers must contain nonnegative integer indexes and cannot be empty."
        raise ValueError(message)
    if len(set(planes)) != len(planes):
        message = "z_layers cannot contain duplicate indexes."
        raise ValueError(message)
    return planes


def validate_options(  # noqa: PLR0913, RUF100 - mirrors the public reader options.
    *,
    vpt_directory: str | Path | None,
    z_layers: Sequence[int] | None,
    region_name: str | None,
    slide_name: str | None,
    include_transcripts: bool,
    include_cell_boundaries: bool,
    include_table: bool,
    include_images: bool,
) -> Selection:
    """Validate the layout options in signature order, before any I/O, and return the selection."""
    if vpt_directory is not None and not isinstance(vpt_directory, str | Path):
        message = f"vpt_directory must be a path or None; found {type(vpt_directory).__name__}."
        raise TypeError(message)

    planes = _validate_z_layers(z_layers)

    for name, parameter in ((region_name, "region_name"), (slide_name, "slide_name")):
        if name is not None:
            validate_element_name(name, parameter=parameter)

    for parameter, value in (
        ("include_transcripts", include_transcripts),
        ("include_cell_boundaries", include_cell_boundaries),
        ("include_table", include_table),
        ("include_images", include_images),
    ):
        if not isinstance(value, bool):
            message = f"{parameter} must be a boolean."
            raise TypeError(message)
    return Selection(include_images, include_cell_boundaries, include_table, include_transcripts, planes)


def _names(root: DatasetRoot, region_name: str | None, slide_name: str | None) -> Names:
    """Use valid caller names or repair the region and slide directory names."""
    region = region_name
    if region is None:
        region = sanitize_element_name(
            root.path.name, source="merscope: the region directory name", parameter="region_name"
        )
    slide = slide_name
    if slide is None:
        slide = sanitize_element_name(
            root.path.parent.name, source="merscope: the slide (parent) directory name", parameter="slide_name"
        )
    names = Names(region, slide)
    # Both parts are valid names, but a slide named "_" starts the prefix with
    # "__", which SpatialData reserves. The fixed suffixes cannot collide.
    validate_element_name(names.dataset_id, parameter="the element-name prefix '<slide_name>_<region_name>'")
    return names


def _file(root: DatasetRoot, name: str, artifact: str) -> ArtifactFile:
    """Resolve one required vendor artifact inside its region or VPT root."""
    return require_metadata_file(root, name, context=ReaderErrorContext("merscope", artifact))


def _vpt_boundaries(analysis: DatasetRoot) -> ArtifactFile:
    """Return the only Cellpose or Watershed boundary file of a VPT analysis."""
    context = ReaderErrorContext("merscope", BOUNDARIES)
    candidates = [
        candidate
        for kind in _SEGMENTATIONS
        if (candidate := optional_metadata_file(analysis, f"{kind}_micron_space.parquet", context=context)) is not None
    ]
    require_unique_path(candidates, context=context, search_path=analysis.path)
    return candidates[0]


def _mosaics(root: DatasetRoot, z_layers: tuple[int, ...] | None) -> Mosaics:
    """Pair every stain with every selected plane; missing or aliased pairs fail."""
    directory = require_metadata_directory(root, "images", context=ReaderErrorContext("merscope", "images"))
    files: dict[tuple[str, int], ArtifactFile] = {}
    for entry in list_directory(directory, context=_MOSAIC_CONTEXT):
        match = _MOSAIC_NAME.fullmatch(entry.name)
        if match is None:
            continue
        key = (match["stain"], int(match["z"]))
        if key in files:
            raise ReaderFormatError(
                context=_MOSAIC_CONTEXT,
                path=directory,
                found=f"multiple files for stain/z {key!r}",
                expected="one file per stain/z pair",
            )
        files[key] = _file(root, f"images/{entry.name}", "mosaic channel")
    if not files:
        raise ReaderFormatError(
            context=_MOSAIC_CONTEXT,
            path=directory,
            found="no supported mosaic files",
            expected="mosaic_<stain>_z<index>.tif files",
        )

    stains = sorted({stain for stain, _ in files})
    selected = tuple(sorted({z for _, z in files})) if z_layers is None else z_layers
    missing = [(stain, z) for z in selected for stain in stains if (stain, z) not in files]
    if missing:
        raise ReaderFormatError(
            context=_MOSAIC_CONTEXT,
            path=directory,
            found=f"missing stain/z pairs {missing!r}",
            expected="every requested plane for every stain",
        )
    layers = tuple(MosaicLayer(z, tuple(MosaicChannel(stain, files[stain, z]) for stain in stains)) for z in selected)
    transform = optional_metadata_file(
        root, _TRANSFORM_FILE, context=ReaderErrorContext("merscope", "micron-to-mosaic transform")
    )
    return Mosaics(layers, transform)


def discover_layout(  # noqa: PLR0913, RUF100 - mirrors the public artifact selections.
    path: str | Path,
    selection: Selection,
    *,
    vpt_directory: str | Path | None = None,
    region_name: str | None = None,
    slide_name: str | None = None,
    path_policy: PathPolicy = "strict",
) -> MerscopeLayout:
    """Resolve the artifacts of the selected products without reading their contents.

    ``region_name`` and ``slide_name`` must already be valid element names.
    """
    root = normalize_dataset_root(path, context=_ROOT_CONTEXT, path_policy=path_policy)
    names = _names(root, region_name, slide_name)

    analysis = root
    if vpt_directory is not None:
        directory = require_caller_directory(root, vpt_directory, context=_VPT_CONTEXT)
        analysis = normalize_dataset_root(directory, context=_VPT_CONTEXT, path_policy=root.path_policy)

    tables = None
    if selection.table:
        tables = TableFiles(_file(analysis, "cell_by_gene.csv", COUNTS), _file(analysis, "cell_metadata.csv", METADATA))

    boundaries = None
    if selection.boundaries:
        boundaries = (
            _file(root, "cell_boundaries.parquet", BOUNDARIES) if analysis is root else _vpt_boundaries(analysis)
        )

    transcripts = None
    if selection.transcripts:
        context = ReaderErrorContext("merscope", TRANSCRIPTS)
        current = (
            analysis is root
            or optional_metadata_file(analysis, "detected_transcripts.csv", context=context) is not None
        )
        source = analysis if current else root
        transcripts = TranscriptSource(
            _file(source, "detected_transcripts.csv", TRANSCRIPTS),
            _file(source, "cell_by_gene.csv", COUNTS),
            "current" if current else "original",
        )

    images = _mosaics(root, selection.z_layers) if selection.images else None
    return MerscopeLayout(names, tables, images, boundaries, transcripts)
