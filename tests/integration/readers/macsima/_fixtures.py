"""Deterministic MACSiQView v0/v1 single-channel OME-TIFFs with explicit oracles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import UUID
from xml.sax.saxutils import escape, quoteattr

import numpy as np
import tifffile
from xarray import DataArray, DataTree

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

type SchemaVersion = Literal[0, 1]
type ScanType = Literal["S", "B", "AF"]

_V0_CYCLE_TYPES = {"S": "AntigenCycle", "B": "BleachCycle"}


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """Describe one exported MACSima channel and its acquisition metadata."""

    marker: str
    cycle: int
    fluorophore: str
    clone: str = ""
    scan: ScanType = "S"
    exposure_ms: str = "100.0"
    shape: tuple[int, int] = (12, 15)
    offset: tuple[int | float, int | float] | None = None
    well: str = "D01"
    roi: int = 1
    file_name: str | None = None
    extra_annotations: Mapping[str, str] = field(default_factory=dict)


V0_CHANNELS = (
    ChannelSpec("CD15", 1, "APC", clone="VIMC6", exposure_ms="2304.0", well="C-1"),
    ChannelSpec("DAPI", 1, "DAPI", exposure_ms="40.0", well="C-1"),
    ChannelSpec("Bcl 2", 2, "FITC", clone="REA872", exposure_ms="96.0", well="C-1"),
    ChannelSpec("CD1c", 2, "PE", clone="REA694", exposure_ms="144.0", well="C-1"),
)
V1_CHANNELS = (
    ChannelSpec("CD3", 1, "APC", clone="REA1151", exposure_ms="1212.52"),
    ChannelSpec("DAPI", 1, "DAPI", exposure_ms="51"),
    ChannelSpec("CD279", 2, "PE", clone="REA1165", exposure_ms="322.12"),
    ChannelSpec("CD66b", 4, "FITC", clone="REA306", exposure_ms="856.68"),
    ChannelSpec("DAPI", 15, "DAPI", scan="B", exposure_ms="51"),
)


def channel_pixels(spec: ChannelSpec, index: int) -> NDArray[np.uint16]:
    """Return unique deterministic values per channel and pixel."""
    rows, columns = spec.shape
    return (np.arange(rows * columns, dtype=np.uint32).reshape(rows, columns) + 1 + 1000 * index).astype(np.uint16)


def _map_annotation(identifier: str, values: Mapping[str, str]) -> str:
    entries = "".join(f"<M K={quoteattr(key)}>{escape(value)}</M>" for key, value in values.items())
    return f'<MapAnnotation ID="{identifier}"><Value>{entries}</Value></MapAnnotation>'


def _plane(offset: tuple[int | float, int | float] | None) -> str:
    if offset is None:
        # The downsampled public exports declare no plane positions.
        return '<Plane TheC="0" TheT="0" TheZ="0"/>'
    y, x = offset
    return (
        f'<Plane TheC="0" TheT="0" TheZ="0" PositionX="{x}" PositionXUnit="pixel" '
        f'PositionY="{y}" PositionYUnit="pixel"/>'
    )


def _annotations(spec: ChannelSpec, version: SchemaVersion) -> tuple[dict[str, str], ...]:
    """Mirror the annotation grouping of the public v0 and v1 exports."""
    if version == 0:
        return (
            {"Instrument name": "SN0126.v1.5.1", "Software version": "v0.15.3"},
            {"Fluorochrome": spec.fluorophore, "Dilution factor": "50"},
            {
                "Exposure time": spec.exposure_ms,
                "Exposure time unit": "ms",
                "Cycle": str(spec.cycle),
                **({"Clone": spec.clone} if spec.clone else {}),
                "Dye": spec.marker,
            },
            {
                "MICS cycle ID": str(spec.cycle),
                "MICS cycle type": _V0_CYCLE_TYPES[spec.scan],
                "ROI ID": str(spec.roi),
                **spec.extra_annotations,
            },
        )
    is_dye = spec.marker == "DAPI"
    return (
        {
            "ExposureTime": spec.exposure_ms,
            "ExposureTimeUnit": "ms",
            "Cycle": str(spec.cycle),
            "Clone": spec.clone,
            "Dye": spec.marker,
        },
        {
            "Cycle": str(spec.cycle),
            "ScanType": spec.scan,
            "RoiId": f"{spec.roi:03d}",
            "MetadataVersion": "v1.1",
            **spec.extra_annotations,
        },
        {"Name": "s-fluor-20x", "WorkingDistance": "6.4"},
        {"Name": "MACSima Imaging System 1.5.1", "SoftwareVersion": "1.1.1"},
        {
            "Clone": spec.clone,
            "Fluorochrome": spec.fluorophore,
            "Biomarker": "" if is_dye else spec.marker,
        },
    )


def channel_ome_xml(  # noqa: PLR0913, RUF100 - Each argument is an independent OME field under test.
    spec: ChannelSpec,
    *,
    version: SchemaVersion,
    file_name: str,
    index: int,
    physical_size: tuple[str, str] = ("0.17", "µm"),
) -> str:
    """Write OME-XML with the same element families as MACSiQView exports."""
    uuid = f"urn:uuid:{UUID(int=index + 1)}"
    size, unit = physical_size
    rows, columns = spec.shape
    reagent = f"{spec.marker}__{spec.clone}" if version == 1 or spec.clone else spec.marker
    annotations = "".join(
        _map_annotation(f"Annotation:{index}:{number}", values)
        for number, values in enumerate(_annotations(spec, version))
    )
    # Instrument and screen IDs are invalid in real exports; ome-types warns and casts them.
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" UUID="{uuid}">'
        f'<Plate ID="Plate:0" Name="01"><Well ID="Well:0" Row="4" Column="1" '
        f"ExternalIdentifier={quoteattr(spec.well)}/></Plate>"
        f'<Screen ID="Screen:"><Reagent ID="Reagent:0" Name={quoteattr(reagent)}/></Screen>'
        '<Instrument ID="Instrument:"><Detector ID="Detector:"/></Instrument>'
        f'<Image ID="Image:0" Name={quoteattr(file_name)}>'
        f'<Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" SizeX="{columns}" SizeY="{rows}" '
        f'SizeZ="1" SizeC="1" SizeT="1" PhysicalSizeX="{size}" PhysicalSizeXUnit="{unit}" '
        f'PhysicalSizeY="{size}" PhysicalSizeYUnit="{unit}">'
        f'<Channel ID="Channel:0:0" SamplesPerPixel="1" Name={quoteattr(spec.marker)}/>'
        f'<TiffData FirstC="0" FirstT="0" FirstZ="0" IFD="0" PlaneCount="1">'
        f"<UUID FileName={quoteattr(file_name)}>{uuid}</UUID></TiffData>"
        f"{_plane(spec.offset)}</Pixels></Image>"
        f"<StructuredAnnotations>{annotations}</StructuredAnnotations></OME>"
    )


def default_file_name(spec: ChannelSpec) -> str:
    """Follow the public export naming scheme; readers must not depend on it."""
    clone = f"_C-{spec.clone}" if spec.clone else ""
    marker = spec.marker.replace(" ", "")
    return (
        f"C-{spec.cycle:03d}_S-000_{spec.scan}_{spec.fluorophore}_R-01_W-{spec.well}_ROI-"
        f"{spec.roi:02d}_A-{marker}{clone}.tif"
    )


def write_channel(  # noqa: PLR0913, RUF100 - Storage and metadata variants are explicit test inputs.
    directory: Path,
    spec: ChannelSpec,
    *,
    version: SchemaVersion,
    index: int,
    physical_size: tuple[str, str] = ("0.17", "µm"),
    stored_name: str | None = None,
) -> NDArray[np.uint16]:
    """Write one-row-strip uncompressed pixels as the public exports do."""
    directory.mkdir(parents=True, exist_ok=True)
    file_name = spec.file_name or default_file_name(spec)
    pixels = channel_pixels(spec, index)
    tifffile.imwrite(
        directory / (stored_name or file_name),
        pixels,
        # MACSiQView stores UTF-8 OME-XML (for example "µm") in the ASCII TIFF tag.
        description=channel_ome_xml(
            spec, version=version, file_name=file_name, index=index, physical_size=physical_size
        ).encode(),
        metadata=None,
        photometric="minisblack",
        rowsperstrip=1,
    )
    return pixels


def write_stack(
    directory: Path,
    channels: Sequence[ChannelSpec],
    *,
    version: SchemaVersion,
    physical_size: tuple[str, str] = ("0.17", "µm"),
    first_index: int = 0,
) -> list[NDArray[np.uint16]]:
    """Write one processed MACSima folder and return pixels in specification order."""
    return [
        write_channel(directory, spec, version=version, index=first_index + index, physical_size=physical_size)
        for index, spec in enumerate(channels)
    ]


def full_resolution(element: DataArray | DataTree) -> DataArray:
    """Return the full-resolution image of a single- or multiscale element."""
    if isinstance(element, DataTree):
        level = element["scale0"]["image"]
        assert isinstance(level, DataArray)
        return level
    return element
