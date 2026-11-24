"""Interpret QPI page descriptions and acquisition profiles.

The PerkinElmer-QPI-ImageDescription XML gives each physical TIFF page its
image type. The first full-resolution page supplies the description version
and acquisition profile. CODEX Instrument Manager (CIM) stores cycle and
channel indices in ExperimentV4 XML. Fusion stores an experiment description
as JSON in ScanProfile, with one-based positions in its well and item lists.

Protocol order can differ from measured page order. Matching uses marker
names and requires equal occurrence counts for repeated names. A spelling
difference caused by punctuation is accepted only for an unambiguous match.
Unmatched channels retain their measured page metadata.

Page exposure is recorded in microseconds and protocol exposure in
milliseconds. Signal units and normalization factors describe the stored
values. The nominal pixel size from XML is recorded separately from the
TIFF calibration used to place the image.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.metadata import parse_json_object

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from spatialdata_io.readers._utils.errors import ReaderErrorContext

type Generation = Literal["fusion1", "fusion2", "cim_qptiff", "qpi_other"]

_XML_DECLARATION = re.compile(r"^\s*<\?xml\b[^>]*\?>")
_XML_DTD = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
_XML_ROOT = "PerkinElmer-QPI-ImageDescription"
_FUSION_VERSION = re.compile(r"Fusion\s+([12])(?:\.[0-9]+)*(?=$|\s)", re.IGNORECASE)
_IMAGE_TYPES = {"FullResolution", "ReducedResolution", "Thumbnail", "Overview", "Label"}


@dataclass(frozen=True, slots=True)
class ProfileChannel:
    """One nonblank cycle/channel record from a source acquisition profile."""

    marker: str
    cycle: int | None
    channel: int | None
    exposure_ms: float | None
    wavelength_nm: float | None = None


@dataclass(frozen=True, slots=True)
class FusionProtocol:
    """Experiment structure normalized from a Fusion profile or XPD object."""

    version: int | None
    cycles: int
    channels: tuple[ProfileChannel, ...]


type PageQuantity = int | float | bool | list[int]


def text(element: ET.Element, tag: str) -> str | None:
    """Return the stripped nonempty text of a single XML field."""
    value = element.findtext(tag)
    return value.strip() or None if value is not None else None


def _nonnegative_integer(value: str | None, *, maximum: int | None = None) -> int | None:
    if value is None or re.fullmatch(r"[0-9]{1,18}", value) is None:
        return None
    number = int(value)
    return number if maximum is None or number <= maximum else None


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _positive(value: str | None) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return {"true": True, "false": False}.get(value.lower())


def _integers(value: str | None, *, maximum: int | None = None) -> list[int] | None:
    if value is None:
        return None
    numbers = [_nonnegative_integer(item, maximum=maximum) for item in re.split(r"[,\s]+", value)]
    return [number for number in numbers if number is not None] if None not in numbers else None


def parse_description(
    description: str, *, page_index: int, context: ReaderErrorContext, path: Path
) -> tuple[ET.Element, str]:
    """Parse one bounded description from a physical QPI page."""
    if _XML_DTD.search(description) is not None:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"an XML document type in IFD {page_index}",
            expected="QPI ImageDescription XML without DTDs or entity declarations",
        )
    try:
        # Some Akoya tags declare UTF-16 even though TIFF stores the XML as ASCII.
        root = ET.fromstring(_XML_DECLARATION.sub("", description, count=1))  # noqa: S314 - DTDs are rejected above.
    except ET.ParseError as error:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"malformed ImageDescription XML in IFD {page_index}",
            expected=f"an XML document with the {_XML_ROOT} root",
        ) from error
    if root.tag != _XML_ROOT or (image_type := text(root, "ImageType")) is None or image_type not in _IMAGE_TYPES:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"an unrecognized QPI ImageDescription root or ImageType in IFD {page_index}",
            expected=f"a {_XML_ROOT} root and a supported ImageType",
        )
    return root, image_type


def description_version(root: ET.Element) -> str | None:
    """Return the numeric version of the page description, when supplied."""
    version = text(root, "DescriptionVersion")
    return version if _nonnegative_integer(version) is not None else None


def scan_generation(root: ET.Element) -> Generation:
    """Recognize the acquisition family from the acquisition software or CIM profile."""
    software = text(root, "AcquisitionSoftware") or ""
    if match := _FUSION_VERSION.match(software):
        return "fusion1" if match.group(1) == "1" else "fusion2"
    if "instrument manager" in software.lower() or root.find("ScanProfile/ExperimentV4") is not None:
        return "cim_qptiff"
    return "qpi_other"


def nominal_pixel_size_um(root: ET.Element) -> float | None:
    """Retrieve the explicitly named Opal/Vectra/inForm or CIM pixel size."""
    size = _positive(text(root, "PixelSizeMicrons") or text(root, "ScanProfile/.//ScanResolution/PixelSizeMicrons"))
    if size is None:
        nanometres = _positive(text(root, "ScanProfile/ExperimentV4/Resolution_nm"))
        size = None if nanometres is None else nanometres / 1000
    return size if size is not None and size > 0 else None


def _object(value: object) -> dict[str, object]:
    # The shared JSON decoder recursively builds dictionaries with string keys.
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def _items(value: object) -> Sequence[object]:
    return value if isinstance(value, list) else ()


def _marker(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    marker = value.strip()
    return marker if marker.lower() not in {"", "--", "blank", "empty"} else None


def fusion_protocol(experiment: dict[str, object]) -> FusionProtocol:
    """Read the ordered well/channel records from a decoded Fusion experiment object."""
    wells = _items(experiment.get("wells"))
    version = experiment.get("formatVersion")
    channels = []
    for cycle_index, well in enumerate(wells, start=1):
        for channel_index, item in enumerate(_items(_object(well).get("items")), start=1):
            record = _object(item)
            marker = _marker(record.get("markerName"))
            if marker is None:
                continue
            exposure = record.get("exposure")
            # The shared JSON boundary checks that decoded integers and floats
            # are representable as finite float64.
            exposure_ms = (
                float(exposure)
                if isinstance(exposure, int | float) and not isinstance(exposure, bool) and exposure >= 0
                else None
            )
            channels.append(ProfileChannel(marker, cycle_index, channel_index, exposure_ms))
    return FusionProtocol(version if type(version) is int and version >= 0 else None, len(wells), tuple(channels))


def profile_channels(root: ET.Element, *, context: ReaderErrorContext, path: Path) -> tuple[ProfileChannel, ...]:
    """Extract the supported Fusion or CIM metadata from the first full-resolution page."""
    if (experiment := root.find("ScanProfile/ExperimentV4")) is not None:
        return tuple(
            ProfileChannel(
                marker,
                _nonnegative_integer(text(cycle, "Index")),
                _nonnegative_integer(text(channel, "Index")),
                _number(text(channel, "ExposureTime_ms")),
                _positive(text(channel, "Wavelength_nm")),
            )
            for cycle in experiment.iterfind("Cycles/Cycle")
            for channel in cycle.iterfind("Channels/Channel")
            if (marker := _marker(text(channel, "MarkerName"))) is not None and text(channel, "MarkerClass") != "Blank"
        )
    content = text(root, "ScanProfile")
    if content is None or not content.startswith("{"):
        return ()
    profile = parse_json_object(content, context=context, path=path)
    return fusion_protocol(_object(profile.get("experimentDescription"))).channels


def _marker_key(name: str) -> str:
    # Some CIM releases sanitize punctuation when the marker is written to an IFD.
    return "".join(character for character in name.casefold() if character.isalnum()) or name


def match_profile(names: Sequence[str], channels: Sequence[ProfileChannel]) -> tuple[ProfileChannel | None, ...]:
    """Assign profile metadata only where the page marker and occurrence are supported."""
    profile: dict[str, dict[str, list[ProfileChannel]]] = defaultdict(lambda: defaultdict(list))
    for channel in channels:
        profile[_marker_key(channel.marker)][channel.marker.casefold()].append(channel)
    pages: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, name in enumerate(names):
        pages[_marker_key(name)][name.casefold()].append(index)

    result: list[ProfileChannel | None] = [None] * len(names)
    for key, page_names in pages.items():
        profile_names = profile.get(key, {})
        for name, indices in page_names.items():
            records = profile_names.get(name, ())
            if not records and len(page_names) == len(profile_names) == 1:
                records = next(iter(profile_names.values()))
            if len(indices) == len(records):
                for index, record in zip(indices, records, strict=True):
                    result[index] = record
    return tuple(result)


def page_quantities(root: ET.Element) -> dict[str, PageQuantity]:
    """Return recognized numeric and flag fields with units in their names."""
    result: dict[str, PageQuantity | None] = {
        "exposure_us": _nonnegative_integer(text(root, "ExposureTime")),
        "exposure_times_us": _integers(text(root, "ExposureTimeArray")),
        "signal_units_code": _nonnegative_integer(text(root, "SignalUnits"), maximum=255),
        "imaging_cycle_zero_based": _nonnegative_integer(text(root, "ImagingCycle")),
        "scale_factor": _positive(text(root, "ScaleFactor")),
        "normalized_units_factor": _positive(text(root, "NormalizedUnitsFactor")),
        "is_unmixed_component": _bool(text(root, "IsUnmixedComponent")),
        "autofluorescence_subtracted": _bool(text(root, "AutofluorescenceSubtracted")),
        "camera_gain": _positive(text(root, "CameraSettings/Gain")),
        "camera_binning": _nonnegative_integer(text(root, "CameraSettings/Binning")) or None,
        "camera_offset_counts": _nonnegative_integer(text(root, "CameraSettings/OffsetCounts")),
        "camera_bit_depth": _nonnegative_integer(text(root, "CameraSettings/BitDepth")) or None,
    }
    color = _integers(text(root, "Color"), maximum=255)
    if color is not None and len(color) == len("rgb"):
        result["color_rgb"] = color
    return {name: value for name, value in result.items() if value is not None}
