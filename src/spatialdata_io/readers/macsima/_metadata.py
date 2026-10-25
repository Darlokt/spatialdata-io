"""Inspect one MACSima channel TIFF and interpret its v0 or v1 OME metadata.

Each processed MACSima channel is a single-plane OME-TIFF. The marker,
acquisition cycle, and acquisition type are stored in OME map annotations
whose keys depend on the MACSiQView software major (0 or 1). The well comes
from the OME plate and, in v0, the marker and clone from the screen reagent.
Pixel size and plane position come from OME ``Pixels``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ome_types.model import UnitsLength

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import inspect_tiff
from spatialdata_io.readers.macsima._schemas import (
    EXPOSURE_UNIT,
    ImageType,
    SchemaMajor,
    normalize_image_type,
    pixel_size_um,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import numpy as np
    from ome_types import OME
    from ome_types.model import Pixels

    from spatialdata_io.readers._utils.image import TiffMetadata
    from spatialdata_io.readers._utils.paths import ArtifactFile

_ARTIFACT = "channel image"
_VERSION_KEYS = ("SoftwareVersion", "Software version")  # v1, v0


@dataclass(frozen=True, slots=True)
class ChannelMetadata:
    """Acquisition annotations of one channel, normalized across schema versions.

    ``name`` is the marker; bleach images already carry the ``_background``
    suffix, so ordering and nuclei matching see the final name. ``exposure_ms``
    is in milliseconds.
    """

    name: str
    cycle: int
    image_type: ImageType
    well: str | None
    roi: int | None
    fluorophore: str | None
    clone: str | None
    exposure_ms: float | None


@dataclass(frozen=True, slots=True)
class ChannelSource:
    """One inspected channel TIFF with normalized metadata and pixel geometry.

    ``tiff`` is the closed inspection result. Pixel graphs are built from it
    without reopening the file for metadata; the OME model never enters a graph.
    """

    tiff: TiffMetadata
    software_version: str
    major: SchemaMajor
    metadata: ChannelMetadata
    pixel_size_um: float
    offset_yx: tuple[int, int]
    shape_yx: tuple[int, int]
    dtype: np.dtype[np.generic]


@dataclass(frozen=True, slots=True)
class _Origin:
    """Error identity of the channel file being interpreted."""

    path: Path
    context: ReaderErrorContext

    def error(self, *, found: str, expected: str) -> ReaderFormatError:
        return ReaderFormatError(context=self.context, path=self.path, found=found, expected=expected)


@dataclass(frozen=True, slots=True)
class _Annotations:
    """Non-blank map-annotation values; repeated keys must agree when read.

    MACSiQView repeats keys across annotation groups. Unread keys may differ
    (``Name`` names both the instrument and the objective), but a key the
    reader interprets must have exactly one value.
    """

    values: Mapping[str, str]
    conflicting: frozenset[str]
    origin: _Origin

    def get(self, key: str) -> str | None:
        if key in self.conflicting:
            raise self.origin.error(found=f"conflicting values for {key!r}", expected="one value per annotation key")
        return self.values.get(key)

    def required(self, key: str) -> str:
        value = self.get(key)
        if value is None:
            raise self.origin.error(found=f"no {key!r} annotation", expected=f"a non-empty {key!r} annotation")
        return value


@dataclass(frozen=True, slots=True)
class _SourceFields:
    """Raw annotation strings selected by one version parser, before validation."""

    name: str | None
    cycle: str | None
    image_type: str
    roi: str | None
    fluorophore: str | None
    clone: str | None
    exposure: str | None
    exposure_unit: str | None


def _annotations(ome: OME, origin: _Origin) -> _Annotations:
    structured = ome.structured_annotations
    map_annotations = structured.map_annotations if structured is not None else []

    values: dict[str, str] = {}
    conflicting: set[str] = set()
    for annotation in map_annotations:
        for entry in annotation.value.ms:
            if entry.k is None or not entry.value.strip():
                continue
            # Keep the first value and record keys repeated with another value.
            if values.setdefault(entry.k, entry.value) != entry.value:
                conflicting.add(entry.k)
    return _Annotations(values, frozenset(conflicting), origin)


def _software_version(annotations: _Annotations) -> tuple[str, SchemaMajor]:
    """Return the version without a ``v`` prefix and its supported major."""
    versions = {
        value.strip().removeprefix("v").removeprefix("V")
        for key in _VERSION_KEYS
        if (value := annotations.get(key)) is not None
    }
    if len(versions) != 1:
        raise annotations.origin.error(
            found=f"software versions {sorted(versions)}",
            expected=f"exactly one version in the {' or '.join(map(repr, _VERSION_KEYS))} annotation",
        )

    version = versions.pop()
    major = version.split(".", 1)[0]
    if major not in {"0", "1"}:
        raise annotations.origin.error(
            found=f"software version {version!r}", expected="MACSiQView major version 0 or 1"
        )
    return version, 0 if major == "0" else 1


def _non_negative_int(text: str, field: str, origin: _Origin) -> int:
    """Parse a non-negative decimal integer such as a cycle or zero-padded ROI ID."""
    if not text.strip().isdecimal():
        raise origin.error(found=f"{field} {text!r}", expected=f"a non-negative decimal {field}")
    return int(text)


def _exposure_ms(text: str | None, unit: str | None, origin: _Origin) -> float | None:
    if text is None:
        return None
    if unit is not None and unit != EXPOSURE_UNIT:
        raise origin.error(found=f"exposure unit {unit!r}", expected=f"exposure in {EXPOSURE_UNIT!r}")

    try:
        exposure = float(text)
    except ValueError as error:
        raise origin.error(found=f"exposure {text!r}", expected="a finite non-negative number") from error
    if not math.isfinite(exposure) or exposure < 0:
        raise origin.error(found=f"exposure {text!r}", expected="a finite non-negative number")
    return exposure


def _well(ome: OME, origin: _Origin) -> str | None:
    """Return the external identifier of the single plate well, if recorded."""
    wells = [well for plate in ome.plates for well in plate.wells]
    if len(wells) > 1:
        raise origin.error(found=f"{len(wells)} plate wells", expected="at most one plate well")
    return (wells[0].external_identifier or None) if wells else None


def _reagent_name(ome: OME, origin: _Origin) -> str:
    """Return the name of the single screen reagent, or an empty string."""
    reagents = [reagent for screen in ome.screens for reagent in screen.reagents]
    if len(reagents) > 1:
        raise origin.error(found=f"{len(reagents)} screen reagents", expected="at most one screen reagent")
    return (reagents[0].name or "") if reagents else ""


def _fields_v0(ome: OME, annotations: _Annotations) -> _SourceFields:
    """Select software major 0 fields; the screen reagent names the marker."""
    # The reagent name is "<marker>__<clone>", or just the marker for dyes.
    marker, _, clone = _reagent_name(ome, annotations.origin).partition("__")
    return _SourceFields(
        name=marker if marker.strip() else None,
        # Old exports lack "Cycle" and only record the MICS cycle ID.
        cycle=annotations.get("Cycle") or annotations.get("MICS cycle ID"),
        image_type=annotations.required("MICS cycle type"),
        roi=annotations.get("ROI ID"),
        fluorophore=annotations.get("Fluorochrome"),
        clone=clone if clone.strip() else None,
        exposure=annotations.get("Exposure time"),
        exposure_unit=annotations.get("Exposure time unit"),
    )


def _fields_v1(annotations: _Annotations) -> _SourceFields:
    """Select software major 1 fields; antibodies are biomarkers and stains are dyes."""
    dye = annotations.get("Dye")
    return _SourceFields(
        name=annotations.get("Biomarker") or dye,
        cycle=annotations.get("Cycle"),
        image_type=annotations.required("ScanType"),
        roi=annotations.get("RoiId"),
        fluorophore=annotations.get("Fluorochrome") or dye,
        clone=annotations.get("Clone"),
        exposure=annotations.get("ExposureTime"),
        exposure_unit=annotations.get("ExposureTimeUnit"),
    )


def _channel_metadata(
    fields: _SourceFields, *, major: SchemaMajor, well: str | None, origin: _Origin
) -> ChannelMetadata:
    """Validate and type the fields selected by one version parser."""
    if fields.name is None:
        raise origin.error(found="no marker name", expected="a marker or dye name")
    if fields.cycle is None:
        raise origin.error(found="no cycle annotation", expected="the acquisition cycle")

    image_type = normalize_image_type(fields.image_type, major=major, context=origin.context, path=origin.path)
    return ChannelMetadata(
        # Bleach images share the marker of their stain image.
        name=f"{fields.name}_background" if image_type == "bleach" else fields.name,
        cycle=_non_negative_int(fields.cycle, "cycle", origin),
        image_type=image_type,
        well=well,
        roi=None if fields.roi is None else _non_negative_int(fields.roi, "ROI ID", origin),
        fluorophore=fields.fluorophore,
        clone=fields.clone,
        exposure_ms=_exposure_ms(fields.exposure, fields.exposure_unit, origin),
    )


def _shape_yx(tiff: TiffMetadata, pixels: Pixels, origin: _Origin) -> tuple[int, int]:
    """Require one two-dimensional plane whose TIFF and OME sizes agree."""
    if (pixels.size_c, pixels.size_z, pixels.size_t) != (1, 1, 1):
        raise origin.error(
            found=f"SizeC/Z/T {(pixels.size_c, pixels.size_z, pixels.size_t)}", expected="one plane per channel file"
        )

    level = tiff.series[0].levels[0]
    sizes = dict(zip(level.axes, level.shape, strict=True))
    if set(sizes) - {"Y", "X"} or (sizes.get("Y"), sizes.get("X")) != (pixels.size_y, pixels.size_x):
        raise origin.error(
            found=f"TIFF axes {level.axes} with shape {level.shape} for OME size {(pixels.size_y, pixels.size_x)}",
            expected="one YX plane matching the OME SizeY and SizeX",
        )
    return sizes["Y"], sizes["X"]


def _isotropic_pixel_size_um(pixels: Pixels, origin: _Origin) -> float:
    """Return the square pixel size in micrometers."""
    size_x = pixel_size_um(
        pixels.physical_size_x, pixels.physical_size_x_unit, context=origin.context, path=origin.path
    )
    size_y = pixel_size_um(
        pixels.physical_size_y, pixels.physical_size_y_unit, context=origin.context, path=origin.path
    )
    if size_x != size_y:
        raise origin.error(found=f"pixel size {size_y} µm (y) by {size_x} µm (x)", expected="square pixels")
    return size_x


def _offset_yx(pixels: Pixels, origin: _Origin) -> tuple[int, int]:
    """Return the plane's pixel offset on the ROI grid; absent positions mean the origin."""
    if len(pixels.planes) > 1:
        raise origin.error(found=f"{len(pixels.planes)} OME planes", expected="at most one plane")
    plane = pixels.planes[0] if pixels.planes else None
    if plane is None or (plane.position_y is None and plane.position_x is None):
        return 0, 0

    y, x = plane.position_y, plane.position_x
    units = (plane.position_y_unit, plane.position_x_unit)
    # MACSiQView v1 records positions in pixels; any other unit would require
    # an unproved stage-to-pixel conversion, and truncation would shift data.
    if y is None or x is None or not (y.is_integer() and x.is_integer()) or units != (UnitsLength.PIXEL,) * 2:
        raise origin.error(
            found=f"plane position {(y, x)} in units {[unit.value for unit in units]}",
            expected="integral PositionY and PositionX in pixel units",
        )
    return int(y), int(x)


def read_channel(path: ArtifactFile) -> ChannelSource | None:
    """Inspect one candidate file once and interpret its MACSima OME metadata.

    Returns
    -------
    ChannelSource | None
        The normalized channel, or ``None`` for a TIFF without OME-XML, which
        is not a MACSima channel export.

    Raises
    ------
    ReaderFormatError
        If OME content violates the supported v0/v1 contract.
    RasterFormatError
        If the TIFF or its OME-XML is malformed.
    OSError
        If the file cannot be read. Inspection errors carry reader context.
    """
    origin = _Origin(path, ReaderErrorContext("macsima", _ARTIFACT))
    try:
        with warnings.catch_warnings():
            # MACSiQView IDs violate OME ID patterns; ome-types casts them and
            # then reports references to the original IDs. Both are benign.
            warnings.filterwarnings("ignore", message=r"Casting invalid \w*ID", category=UserWarning)
            warnings.filterwarnings("ignore", message=r"Reference to unknown ID", category=UserWarning)
            tiff = inspect_tiff(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=origin.context, path=path)
        raise

    ome = tiff.ome
    if ome is None:
        return None
    if len(ome.images) != 1:
        raise origin.error(found=f"{len(ome.images)} OME images", expected="one image per channel file")
    pixels = ome.images[0].pixels

    annotations = _annotations(ome, origin)
    version, major = _software_version(annotations)
    # Errors after version detection also name the detected software version.
    origin = _Origin(path, ReaderErrorContext("macsima", _ARTIFACT, detected_version=version))
    annotations = replace(annotations, origin=origin)
    fields = _fields_v0(ome, annotations) if major == 0 else _fields_v1(annotations)

    return ChannelSource(
        tiff=tiff,
        software_version=version,
        major=major,
        metadata=_channel_metadata(fields, major=major, well=_well(ome, origin), origin=origin),
        pixel_size_um=_isotropic_pixel_size_um(pixels, origin),
        offset_yx=_offset_yx(pixels, origin),
        shape_yx=_shape_yx(tiff, pixels, origin),
        dtype=tiff.series[0].levels[0].dtype,
    )
