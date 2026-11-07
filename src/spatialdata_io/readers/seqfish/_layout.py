"""Discover the ROIs of a Spatial Genomics seqFISH export and the artifacts each selected product needs.

A Spatial Genomics export directory holds one flat group of files per region
of interest (ROI), each named ``<ROI>_<artifact>``::

    dataset/
    ├── Roi1_CellxGene.csv          # required: cell-by-feature counts
    ├── Roi1_CellCoordinates.csv    # required: cell ID, area (µm²), centroid (µm)
    ├── Roi1_DAPI.tiff              # OME-TIFF; images, labels and polygons
    ├── Roi1_Segmentation.tiff      # integer cell labels; labels
    ├── Roi1_TranscriptList.csv     # decoded transcripts (µm); points
    ├── Roi1_Boundaries.geojson     # cell polygons (DAPI pixels); polygons
    ├── Roi2_CellxGene.csv
    └── ...

Column schemas are described in :mod:`._schemas`. The export has no version
marker, so no format version is detected.

- A ROI is every prefix of a file ending in ``_CellxGene.csv`` or
  ``_CellCoordinates.csv``. ROIs are read in lexical prefix order, or in the
  caller's ``rois`` order, which names exact prefixes.
- Each selected ROI needs both tables. The other artifacts are required only
  when a selected product reads them: the DAPI image calibrates images,
  labels and polygons even when the image itself is not returned, while
  tables, circles and transcripts are already in µm and need no raster.
- The ROI name prefixes element names (``<ROI>_DAPI``, ``table_<ROI>``, ...)
  and names the ROI's frame when several ROIs are read. A prefix that is not
  a valid SpatialData element name is repaired with a warning (``Roi 1``
  becomes ``Roi_1``). Repaired names of the selected ROIs must stay distinct
  ignoring letter case, because SpatialData compares element names that way
  and Zarr stores may sit on case-insensitive filesystems.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.geometry import validate_invalid_geometry_policy
from spatialdata_io.readers._utils.names import sanitize_element_name
from spatialdata_io.readers._utils.paths import (
    ArtifactFile,
    DatasetRoot,
    list_directory,
    normalize_dataset_root,
    require_metadata_file,
)

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.geometry import InvalidGeometryPolicy
    from spatialdata_io.readers._utils.paths import PathPolicy

_CONTEXT = ReaderErrorContext("seqfish", "ROI layout")
_TABLE_SUFFIXES = ("_CellxGene.csv", "_CellCoordinates.csv")


@dataclass(frozen=True, slots=True)
class Selection:
    """Validated product choices and the caller's ROI prefixes, owned as a tuple."""

    images: bool
    labels: bool
    transcripts: bool
    shapes: Literal["none", "polygons", "circles", "both"]
    invalid_geometries: InvalidGeometryPolicy
    rois: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class PixelLayout:
    """The DAPI calibration source and the requested rasters and polygons on its pixel grid."""

    dapi: ArtifactFile
    labels: ArtifactFile | None
    boundaries: ArtifactFile | None


@dataclass(frozen=True, slots=True)
class RoiLayout:
    """The validated artifacts of one ROI under its element name."""

    name: str
    counts: ArtifactFile
    cells: ArtifactFile
    pixels: PixelLayout | None
    transcripts: ArtifactFile | None


def make_selection(  # noqa: PLR0913, RUF100 - Keep supported product choices explicit.
    *,
    include_images: bool,
    include_labels: bool,
    include_transcripts: bool,
    include_shapes: bool,
    shape_geometry: Literal["polygons", "circles", "both"],
    invalid_geometries: Literal["keep", "drop", "raise"],
    rois: Sequence[str] | None,
) -> Selection:
    """Validate public choices in signature order, before discovery, and copy borrowed sequences."""
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

    policy = validate_invalid_geometry_policy(invalid_geometries)

    selected = None
    if rois is not None:
        if isinstance(rois, str | bytes) or not isinstance(rois, Sequence):
            message = "rois must be a sequence of ROI names, not a string."
            raise TypeError(message)
        selected = tuple(rois)
        if not all(isinstance(name, str) for name in selected):
            message = "rois must contain only strings."
            raise TypeError(message)
        if not selected:
            message = "rois must name at least one ROI; pass None to read all ROIs."
            raise ValueError(message)
        if len(set(selected)) != len(selected):
            message = "rois must not contain duplicate ROI names."
            raise ValueError(message)

    return Selection(
        images=include_images,
        labels=include_labels,
        transcripts=include_transcripts,
        shapes=shape_geometry if include_shapes else "none",
        invalid_geometries=policy,
        rois=selected,
    )


def discover_layout(
    path: str | Path, selection: Selection, *, path_policy: PathPolicy = "strict"
) -> tuple[RoiLayout, ...]:
    """List the dataset directory once and resolve the selected ROIs' required artifacts."""
    root = normalize_dataset_root(path, context=_CONTEXT, path_policy=path_policy)
    entries = list_directory(root.path, context=_CONTEXT)
    prefixes = sorted(
        {
            entry.name.removesuffix(suffix)
            for entry in entries
            for suffix in _TABLE_SUFFIXES
            if entry.name.endswith(suffix)
        }
    )
    if not prefixes:
        raise ReaderFormatError(
            context=_CONTEXT,
            path=root.path,
            found="no ROI files",
            expected="at least one <ROI>_CellxGene.csv or <ROI>_CellCoordinates.csv",
        )

    requested = selection.rois if selection.rois is not None else tuple(prefixes)
    unknown = sorted(set(requested).difference(prefixes))
    if unknown:
        raise ReaderFormatError(
            context=_CONTEXT,
            path=root.path,
            found=f"unknown ROIs {unknown!r}",
            expected=f"ROIs in {prefixes!r}",
        )

    names = _element_names(requested, root)
    return tuple(_roi_layout(root, prefix, name, selection) for prefix, name in zip(requested, names, strict=True))


def _element_names(prefixes: Sequence[str], root: DatasetRoot) -> list[str]:
    """Repair invalid ROI prefixes into element names that stay distinct ignoring case."""
    names = [
        sanitize_element_name(prefix, source="seqfish: the ROI file prefix", parameter=None) for prefix in prefixes
    ]
    folded = Counter(name.lower() for name in names)
    conflicts = [prefix for prefix, name in zip(prefixes, names, strict=True) if folded[name.lower()] > 1]
    if conflicts:
        raise ReaderFormatError(
            context=_CONTEXT,
            path=root.path,
            found=f"ROIs {conflicts!r} whose element names are equal ignoring case",
            expected="ROI element names that are distinct ignoring case; select one of them with rois",
        )
    return names


def _roi_layout(root: DatasetRoot, prefix: str, name: str, selection: Selection) -> RoiLayout:
    """Resolve the ROI's tables and the artifacts of its selected products; none may be missing."""

    def artifact(kind: str, extension: str) -> ArtifactFile:
        return require_metadata_file(root, f"{prefix}_{kind}.{extension}", context=ReaderErrorContext("seqfish", kind))

    counts = artifact("CellxGene", "csv")
    cells = artifact("CellCoordinates", "csv")
    polygons = selection.shapes in {"polygons", "both"}
    pixels = None
    if selection.images or selection.labels or polygons:
        pixels = PixelLayout(
            dapi=artifact("DAPI", "tiff"),
            labels=artifact("Segmentation", "tiff") if selection.labels else None,
            boundaries=artifact("Boundaries", "geojson") if polygons else None,
        )
    transcripts = artifact("TranscriptList", "csv") if selection.transcripts else None
    return RoiLayout(name, counts, cells, pixels, transcripts)
