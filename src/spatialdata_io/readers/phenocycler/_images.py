"""Build a lazy channel-first image from the measured pages of a QPI scan.

The first TIFF Software tag identifies a PerkinElmer-QPI file. Each physical
image file directory (IFD) declares its role in the ImageDescription XML.
A thumbnail can have the same dimensions as a measured plane. Only pages
labelled ``FullResolution`` contribute pixels, in physical IFD order.

Monochrome planes must share a top-left YX grid and dtype. Each channel takes
its name from ``Biomarker``, then the optical band ``Name``, then a zero-based
``channel_<index>`` fallback. A single RGB page supplies three sample channels.
The output preserves the stored values and dtype. Display-name overrides
leave the source biomarker, band and page index available as channel metadata.

Inspection converts matching positive X and Y resolutions in pixels per
centimetre to micrometres per pixel. The 96 dpi display convention leaves
the scan uncalibrated. The shared TIFF reader opens each selected page on
demand, and the raster model builds the requested chunks and pyramid levels.

The page roles, resolution convention and signal fields are specified in
Appendix B of the Akoya Phenochart User Guide, version 2.2.0.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import dask.array as da
from spatialdata.models import Image2DModel

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    normalize_cyx_image,
    normalize_raster_axes,
    raster_channel_names,
    read_tiff,
)
from spatialdata_io.readers.phenocycler._metadata import (
    description_version,
    match_profile,
    nominal_pixel_size_um,
    page_quantities,
    parse_description,
    profile_channels,
    scan_generation,
    text,
)

if TYPE_CHECKING:
    import xml.etree.ElementTree as ET
    from collections.abc import Mapping, Sequence

    import numpy as np
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.image import RasterOptions, TiffMetadata, TiffPageMetadata
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.phenocycler._metadata import Generation

_CENTIMETRE = 3  # TIFF ResolutionUnit
_UM_PER_CM = 10_000
_TIFF_POSITION_TOLERANCE_UM = 1e-3
_TIFF_RESOLUTION_RTOL = 1e-6
_TIFF_RGB_AXES = {"S", "Y", "X"}
_TIFF_MONO_AXES = {"Y", "X"}
_BLACK_IS_ZERO = 1  # TIFF PhotometricInterpretation
_PATH_LIKE_TEXT = re.compile(r"^[/\\~]|^(?:[A-Z]:[/\\]|file:)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class QpiChannel:
    """One output channel and its allowlisted quantitative information."""

    channel_id: str
    source_name: str
    page_index: int
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class QpiScan:
    """Closed metadata for a single QPI raster."""

    tiff: TiffMetadata
    pages: tuple[int, ...]
    shape: tuple[int, int]
    dtype: np.dtype[np.generic]
    generation: Generation
    format_version: str | None
    software: str | None
    slide_id: str | None
    pixel_size_um: tuple[float, float] | None
    nominal_pixel_size_um: float | None
    stage_position_um: tuple[float, float] | None
    channels: tuple[QpiChannel, ...]

    @property
    def channel_ids(self) -> tuple[str, ...]:
        """The source-derived channel coordinates in pixel array order."""
        return tuple(channel.channel_id for channel in self.channels)


def matching_pixel_sizes(first: tuple[float, float], second: tuple[float, float]) -> bool:
    """Compare scales while allowing a TIFF rational encoding to round."""
    return all(math.isclose(a, b, rel_tol=_TIFF_RESOLUTION_RTOL) for a, b in zip(first, second, strict=True))


def matching_stage_positions(first: tuple[float, float], second: tuple[float, float]) -> bool:
    """Compare micrometre positions while allowing subnanometre rounding."""
    return all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=_TIFF_POSITION_TOLERANCE_UM) for a, b in zip(first, second, strict=True)
    )


def _tiff_pixel_size(
    page: TiffPageMetadata, *, context: ReaderErrorContext, path: ArtifactFile
) -> tuple[float, float] | None:
    # In QPI, the inch unit (typically 96 px/in) is an uncalibrated display default.
    if page.resolution_unit != _CENTIMETRE:
        return None
    x, y = page.x_resolution, page.y_resolution
    if x is None or y is None or min(*x, *y) <= 0:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"invalid XResolution/YResolution in IFD {page.index}",
            expected="two positive pixels-per-centimetre TIFF rationals",
        )
    scales = (_UM_PER_CM * x[1] / x[0], _UM_PER_CM * y[1] / y[0])
    if not math.isclose(*scales, rel_tol=_TIFF_RESOLUTION_RTOL):
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"different XResolution and YResolution in IFD {page.index}",
            expected="matching X and Y pixel sizes for a square-pixel QPI raster",
        )
    return scales


def _tiff_stage_position(
    page: TiffPageMetadata, *, context: ReaderErrorContext, path: ArtifactFile
) -> tuple[float, float] | None:
    if page.resolution_unit != _CENTIMETRE or page.x_position is None or page.y_position is None:
        return None
    x, y = page.x_position, page.y_position
    if x[1] <= 0 or y[1] <= 0:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"invalid stage position in IFD {page.index}",
            expected="XPosition and YPosition TIFF rationals with positive denominators",
        )
    return _UM_PER_CM * x[0] / x[1], _UM_PER_CM * y[0] / y[1]


def _scan_grid(
    pages: Sequence[TiffPageMetadata], *, context: ReaderErrorContext, path: ArtifactFile
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    sizes = [
        (page.index, size) for page in pages if (size := _tiff_pixel_size(page, context=context, path=path)) is not None
    ]
    positions = [
        (page.index, position)
        for page in pages
        if (position := _tiff_stage_position(page, context=context, path=path)) is not None
    ]
    for tag, measures, agree in (
        ("pixel sizes", sizes, matching_pixel_sizes),
        ("stage positions", positions, matching_stage_positions),
    ):
        if measures and any(not agree(value, measures[0][1]) for _, value in measures[1:]):
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"conflicting {tag} in full-resolution IFDs {measures}",
                expected="a single registered grid for the full-resolution channels",
            )
    return sizes[0][1] if sizes else None, positions[0][1] if positions else None


def _spatial_shape(page: TiffPageMetadata) -> tuple[int, int]:
    axes = tuple(axis.upper() for axis in page.axes)
    return page.shape[axes.index("Y")], page.shape[axes.index("X")]


def _validate_pages(
    pages: Sequence[TiffPageMetadata], *, context: ReaderErrorContext, path: ArtifactFile
) -> tuple[int, int]:
    first = pages[0]
    for page in pages:
        axes = {axis.upper() for axis in page.axes}
        color_page = (
            len(pages) == 1
            and page.color == "rgb"
            and axes == _TIFF_RGB_AXES
            and page.samples_per_pixel == len(page.color)
        )
        mono_page = page.samples_per_pixel == 1 and axes == _TIFF_MONO_AXES and page.photometric == _BLACK_IS_ZERO
        if not (color_page or mono_page) or page.orientation not in {None, 1} or page.dtype.kind not in "iuf":
            raise ReaderFormatError(
                context=context,
                path=path,
                found=(
                    f"IFD {page.index} with axes {page.axes}, {page.samples_per_pixel} samples, "
                    f"dtype {page.dtype}, photometric {page.photometric}, and TIFF orientation {page.orientation}"
                ),
                expected="one top-left RGB page or BlackIsZero top-left YX pages with real numeric pixels",
            )
        if page.dtype != first.dtype or _spatial_shape(page) != _spatial_shape(first):
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"mismatched shape or dtype in full-resolution IFD {page.index}",
                expected="matching spatial shapes and dtypes for all full-resolution channels",
            )
    return _spatial_shape(first)


def _channel_ids(names: Sequence[str]) -> tuple[str, ...]:
    taken = set(names)
    seen: set[str] = set()
    next_suffix: dict[str, int] = {}
    ids = []
    for name in names:
        channel_id = name
        if name in seen:
            suffix = next_suffix.get(name, 2)
            while (channel_id := f"{name} ({suffix})") in taken:
                suffix += 1
            next_suffix[name] = suffix + 1
            taken.add(channel_id)
        seen.add(name)
        ids.append(channel_id)
    return tuple(ids)


def _source_text(value: str | None) -> str | None:
    return value if value is not None and _PATH_LIKE_TEXT.match(value) is None else None


def _sample_value(value: float | tuple[int | float, ...] | None) -> int | float | list[int | float] | None:
    if isinstance(value, tuple):
        if not all(math.isfinite(item) for item in value):
            return None
        return value[0] if len(value) == 1 else list(value)
    return value if value is not None and math.isfinite(value) else None


def _channels(
    pages: Sequence[TiffPageMetadata],
    descriptions: Sequence[ET.Element],
    *,
    context: ReaderErrorContext,
    path: ArtifactFile,
) -> tuple[QpiChannel, ...]:
    color_names = raster_channel_names(pages[0].color, samples=True, ome=None)
    if color_names is not None:
        pages = [pages[0]] * len(color_names)
        descriptions = [descriptions[0]] * len(color_names)
        names = color_names
    else:
        names = [
            _source_text(text(root, "Biomarker")) or _source_text(text(root, "Name")) or f"channel_{index}"
            for index, root in enumerate(descriptions)
        ]
    records = match_profile(names, profile_channels(descriptions[0], context=context, path=path))
    result = []
    for channel_id, name, page, root, profile in zip(
        _channel_ids(names), names, pages, descriptions, records, strict=True
    ):
        details: dict[str, object] = {
            "biomarker": _source_text(text(root, "Biomarker")),
            "band_name": _source_text(text(root, "Name")),
            "sample_min": _sample_value(page.min_sample_value),
            "sample_max": _sample_value(page.max_sample_value),
            **page_quantities(root),
        }
        if profile is not None:
            details.update(
                {
                    "profile_cycle": profile.cycle,
                    "profile_channel": profile.channel,
                    "profile_exposure_ms": profile.exposure_ms,
                    "profile_wavelength_nm": profile.wavelength_nm,
                }
            )
        result.append(
            QpiChannel(channel_id, name, page.index, {key: val for key, val in details.items() if val is not None})
        )
    return tuple(result)


def _full_resolution(
    metadata: TiffMetadata, *, context: ReaderErrorContext
) -> tuple[tuple[TiffPageMetadata, ...], tuple[ET.Element, ...], str | None]:
    pages = metadata.pages or ()
    if not pages or not pages[0].software.startswith("PerkinElmer-QPI"):
        raise ReaderFormatError(
            context=context,
            path=metadata.path,
            found="a TIFF without the QPI Software signature",
            expected="a TIFF whose first Software tag begins with PerkinElmer-QPI",
        )
    full_pages: list[TiffPageMetadata] = []
    full_descriptions: list[ET.Element] = []
    version = None
    for page in pages:
        root, image_type = parse_description(
            page.description, page_index=page.index, context=context, path=metadata.path
        )
        if image_type == "FullResolution":
            if not full_pages:
                version = description_version(root)
                context = replace(context, detected_version=version)
            full_pages.append(page)
            full_descriptions.append(root)
    if not full_pages:
        raise ReaderFormatError(
            context=context,
            path=metadata.path,
            found="no FullResolution pages",
            expected="at least one IFD declaring ImageType FullResolution",
        )
    return tuple(full_pages), tuple(full_descriptions), version


def _slide_id(descriptions: Sequence[ET.Element], *, context: ReaderErrorContext, path: ArtifactFile) -> str | None:
    # Unknown values cannot establish that separately saved scans share a slide.
    slides = {
        slide
        for root in descriptions
        if (slide := text(root, "SlideID")) is not None and slide.casefold() not in {"unknown", "none", "n/a"}
    }
    if len(slides) > 1:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="conflicting SlideID fields in the full-resolution pages",
            expected="one slide per full-resolution scan",
        )
    return next(iter(slides), None)


def inspect_scan(path: ArtifactFile, *, context: ReaderErrorContext) -> QpiScan:
    """Inspect the physical IFDs, grid and channel identities without reading pixels."""
    metadata = inspect_tiff(path, context=context, include_pages=True)
    pages, descriptions, version = _full_resolution(metadata, context=context)
    baseline = descriptions[0]
    context = replace(context, detected_version=version)
    shape = _validate_pages(pages, context=context, path=path)
    calibrated, stage = _scan_grid(pages, context=context, path=path)
    nominal = nominal_pixel_size_um(baseline)
    return QpiScan(
        tiff=metadata,
        pages=tuple(page.index for page in pages),
        shape=shape,
        dtype=pages[0].dtype,
        generation=scan_generation(baseline),
        format_version=version,
        software=_source_text(text(baseline, "AcquisitionSoftware")),
        slide_id=_slide_id(descriptions, context=context, path=path),
        pixel_size_um=calibrated,
        nominal_pixel_size_um=nominal,
        stage_position_um=stage,
        channels=_channels(pages, descriptions, context=context, path=path),
    )


def _names(scan: QpiScan, names: Sequence[str] | None) -> list[str]:
    if names is None:
        return list(scan.channel_ids)
    if len(names) != len(scan.channels):
        message = (
            f"channel_names has {len(names)} entries, but {scan.tiff.path.name!r} has {len(scan.channels)} channels."
        )
        raise ValueError(message)
    return list(names)


def _read_pixels(scan: QpiScan, context: ReaderErrorContext) -> da.Array:
    # Read the XML-selected IFDs directly. A series read would rebuild the
    # complete QPI series and pyramid index for every requested source tile.
    rasters = (read_tiff(scan.tiff, page=page, context=context) for page in scan.pages)
    if len(scan.pages) == 1:
        return normalize_cyx_image(next(rasters), None, tiff=None, context=context, path=scan.tiff.path)[0]
    planes = [normalize_raster_axes(raster.data, raster.axes, target_axes=("y", "x")) for raster in rasters]
    return da.stack(planes, axis=0)


def read_scan(
    scan: QpiScan,
    *,
    options: RasterOptions,
    channel_names: Sequence[str] | None,
    transformations: Mapping[str, BaseTransformation],
    context: ReaderErrorContext,
) -> DataArray | DataTree:
    """Construct one lazy QPI CYX image and optional generated pyramid."""
    names = _names(scan, channel_names)
    pixels = _read_pixels(scan, context)
    options.validate_image_pyramid(pixels, element=f"QPI scan {scan.tiff.path.name!r}")
    return Image2DModel.parse(
        pixels,
        dims=("c", "y", "x"),
        c_coords=names,
        transformations=dict(transformations),
        scale_factors=options.model_scale_factors,
        chunks=options.image_chunks(),
    )


def scan_attributes(scan: QpiScan, channel_names: Sequence[str] | None = None) -> dict[str, object]:
    """Build a portable summary of the scan, independent of filenames or raw XML."""
    names = _names(scan, channel_names)
    return {
        "generation": scan.generation,
        "description_version": scan.format_version,
        "software": scan.software,
        "pixel_size_um": None if scan.pixel_size_um is None else list(scan.pixel_size_um),
        "nominal_pixel_size_um": scan.nominal_pixel_size_um,
        "stage_position_um": None if scan.stage_position_um is None else list(scan.stage_position_um),
        "channels": [
            {
                "id": channel.channel_id,
                "source_name": channel.source_name,
                "name": name,
                "page_index": channel.page_index,
                **channel.metadata,
            }
            for channel, name in zip(scan.channels, names, strict=True)
        ],
    }
