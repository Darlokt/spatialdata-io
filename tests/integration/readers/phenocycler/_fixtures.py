"""Deterministic TIFF writers for Akoya QPI format fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

import numpy as np
import tifffile

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray


FUSION_PROFILE = escape(
    json.dumps(
        {
            "experimentDescription": {
                "formatVersion": 2,
                "operatorName": "/private/operator",
                "wells": [
                    {"items": [{"markerName": "DAPI", "exposure": 88}, {"markerName": "--"}]},
                    {"items": [{"markerName": "CD3", "exposure": 120}]},
                ],
            }
        }
    )
)

CIM_PROFILE = (
    "<ExperimentV4><Resolution_nm>500</Resolution_nm><Cycles>"
    "<Cycle><Index>1</Index><Channels>"
    "<Channel><Index>1</Index><MarkerName>DAPI</MarkerName><ExposureTime_ms>20</ExposureTime_ms>"
    "<Wavelength_nm>458</Wavelength_nm></Channel>"
    "<Channel><Index>2</Index><MarkerName>CD3</MarkerName><ExposureTime_ms>100</ExposureTime_ms>"
    "<Wavelength_nm>520</Wavelength_nm></Channel></Channels></Cycle></Cycles></ExperimentV4>"
)

OPAL_PROFILE = "<OpalScan><ScanResolution><PixelSizeMicrons>0.5</PixelSizeMicrons></ScanResolution></OpalScan>"


@dataclass(frozen=True, slots=True)
class QpiPage:
    pixels: NDArray[np.generic]
    image_type: str = "FullResolution"
    name: str | None = None
    marker: str | None = None
    scan_profile: str | None = None
    extra_xml: str = ""
    resolution: tuple[float, float] = (20000, 20000)
    resolution_unit: str = "CENTIMETER"
    stage_position: tuple[float, float] | None = None
    minimum: float | None = None
    maximum: float | None = None
    description: str | None = field(default=None, repr=False)


def write_pages(  # noqa: PLR0913 - each fixture parameter selects a distinct format fact.
    path: Path,
    pages: Sequence[QpiPage],
    *,
    software: str = "Fusion 2.3.1",
    tiff_software: str = "PerkinElmer-QPI",
    slide_id: str = "sample",
    scan_id: str = "same-acquisition",
    version: str = "5",
    tiled: bool = True,
    compression: str | None = "deflate",
) -> None:
    with tifffile.TiffWriter(path, bigtiff=path.suffix == ".qptiff") as writer:
        for page in pages:
            rgb = page.pixels.ndim == 3
            fields = {
                "DescriptionVersion": version,
                "AcquisitionSoftware": software,
                "ImageType": page.image_type,
                "Identifier": scan_id,
                "SlideID": slide_id,
                "OperatorName": "/private/operator",
                "ComputerName": "private-host",
                "SampleDescription": "/private/sample",
                "Name": page.name,
                "Biomarker": page.marker,
            }
            xml = "".join(f"<{key}>{escape(value)}</{key}>" for key, value in fields.items() if value is not None)
            if page.scan_profile is not None:
                xml += f"<ScanProfile>{page.scan_profile}</ScanProfile>"
            description = (
                '<?xml version="1.0" encoding="utf-16"?>'
                f"<PerkinElmer-QPI-ImageDescription>{xml}{page.extra_xml}</PerkinElmer-QPI-ImageDescription>"
            )
            tags: list[tuple[int, str, int, tuple[int, int] | float, bool]] = []
            if page.stage_position is not None:
                tags.extend(
                    (code, "2I", 1, (round(position * 1e6), 1_000_000), False)
                    for code, position in zip((286, 287), page.stage_position, strict=True)
                )
            for code, bound in ((340, page.minimum), (341, page.maximum)):
                if bound is not None:
                    tags.append((code, "f", 1, bound, False))
            writer.write(
                page.pixels,
                metadata=None,
                description=description if page.description is None else page.description,
                software=tiff_software,
                photometric="rgb" if rgb else "minisblack",
                resolution=page.resolution,
                resolutionunit=page.resolution_unit,
                tile=(16, 16) if tiled else None,
                rowsperstrip=None if tiled else 4,
                compression=compression,
                subfiletype=int(page.image_type != "FullResolution"),
                extratags=tags,
            )


def write_qptiff(  # noqa: PLR0913 - each fixture parameter selects a distinct format fact.
    path: Path,
    *,
    software: str = "Fusion 2.3.1",
    scan_profile: str | None = None,
    markers: Sequence[str | None] = ("DAPI", "CD3"),
    names: Sequence[str] | None = None,
    dtype: str = "uint16",
    rgb: bool = False,
    shape: tuple[int, int] = (32, 48),
    resolution: tuple[float, float] = (20000, 20000),
    resolution_unit: str = "CENTIMETER",
    stage_position: tuple[float, float] | None = None,
    slide_id: str = "sample",
    scan_id: str = "same-acquisition",
    extra_xml: str = "<ExposureTime>12000</ExposureTime><SignalUnits>64</SignalUnits>",
    tiled: bool = True,
    compression: str | None = "deflate",
) -> NDArray[np.generic]:
    """Write channels with intervening thumbnail and reduced-spectrum IFDs."""
    full_shape = (*shape, 3) if rgb else (len(markers), *shape)
    data = (np.arange(np.prod(full_shape)).reshape(full_shape) % 246).astype(dtype)
    bands = (
        [(None, None, data)]
        if rgb
        else [
            (marker, names[index] if names is not None else marker, data[index]) for index, marker in enumerate(markers)
        ]
    )
    pages = [
        QpiPage(
            pixels,
            marker=marker,
            name=name,
            scan_profile=scan_profile if index == 0 else None,
            extra_xml=extra_xml,
            resolution=resolution,
            resolution_unit=resolution_unit,
            stage_position=stage_position if index == 0 else None,
        )
        for index, (marker, name, pixels) in enumerate(bands)
    ]
    thumbnail = QpiPage(np.full((8, 12, 3), 22, dtype=np.uint8), "Thumbnail")
    # A same-dtype mono reduced page between full-resolution channels challenges
    # TIFF series grouping. Page XML defines membership and order.
    reduced = QpiPage(data[::2, ::2] if rgb else data[0, ::2, ::2], "ReducedResolution")
    overview = QpiPage(np.full((7, 9, 3), 33, dtype=np.uint8), "Overview")
    label = QpiPage(np.full((5, 7, 3), 44, dtype=np.uint8), "Label")
    write_pages(
        path,
        [*pages[:1], thumbnail, reduced, *pages[1:], overview, label],
        software=software,
        slide_id=slide_id,
        scan_id=scan_id,
        tiled=tiled,
        compression=compression,
    )
    return np.moveaxis(data, -1, 0) if rgb else data
