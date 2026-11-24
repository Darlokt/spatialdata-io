"""Read the Visium tissue images lazily as CYX image elements.

Space Ranger writes two downsampled copies of the tissue image it registered,
both optional for this reader; the full-resolution image itself is an input to
Space Ranger and is read only when the caller passes it::

    spatial/tissue_hires_image.png   full-resolution x tissue_hires_scalef
    spatial/tissue_lowres_image.png  full-resolution x tissue_lowres_scalef
    fullres_image_file               TIFF, OME-TIFF, BigTIFF/BTF, PNG or JPEG

Axes and channels::

    source axes    TIFF: file-declared (``YX``, ``CYX``, ``YXS``, ...) or
                   ``data_axes``; PNG and JPEG: ``YXS``
    samples (S)    interleaved samples become channels
    other axes     singleton dimensions are dropped; any other extra axis fails
    channels       ``r, g, b(, a)`` for samples the file declares RGB(A);
                   otherwise OME channel names when all are nonempty and
                   unique, else OME channel IDs; positional integers without
                   either declaration

RGB(A) is declared by TIFF ``PhotometricInterpretation`` RGB (alpha through
``ExtraSamples``) or JPEG-compressed YCbCr, PNG colour types 2, 3 and 6 (alpha
through colour type 6 or ``tRNS``), and three-component JPEG. The channel
count or dtype never makes an image RGB, so a four-channel fluorescence TIFF
keeps its OME names or positions. OME names apply when the image's channel
axis is the OME channel (or samples) axis, or when the file has a single
channel; a ``data_axes`` reinterpretation of another axis as channels gets
positions.

Axes the file declares that cannot become ``c, y, x`` are malformed content and
raise ``ReaderFormatError``; axes from ``data_axes`` that do not fit the file
raise ``ValueError`` with a note naming the image.

A TIFF has one series unless ``tiff_series`` selects one of several; the
full-resolution level of that series is read. Only the full-resolution image
gets generated pyramid levels; the PNGs are already downsampled.
The original image can calibrate its full-resolution grid when OME pixel
sizes describe both spatial axes and agree after conversion to micrometres.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from spatialdata.models import Image2DModel
from spatialdata.transformations import BaseTransformation, Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    make_raster_options,
    normalize_cyx_image,
    ome_pixel_sizes_um,
    read_jpeg,
    read_png,
    read_tiff,
    select_tiff_series,
    validate_data_axes,
    validate_tiff_index,
)
from spatialdata_io.readers._utils.image._ome import PIXEL_SIZE_RTOL
from spatialdata_io.readers.visium._geometry import isotropic_scale

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dask.array import Array
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.visium._geometry import VisiumFrames
    from spatialdata_io.readers.visium._layout import VisiumLayout

type ImageKind = Literal["tiff", "png", "jpeg"]

_IMAGE_KINDS: dict[str, ImageKind] = {
    ".btf": "tiff",
    ".tif": "tiff",
    ".tiff": "tiff",
    ".png": "png",
    ".jpeg": "jpeg",
    ".jpg": "jpeg",
}
_IMAGE_AXES = ("c", "y", "x")


@dataclass(frozen=True, slots=True)
class ImageOptions:
    """Validated full-resolution selection and the shared pyramid and chunk options."""

    raster: RasterOptions
    data_axes: str | None
    tiff_series: int | None


@dataclass(frozen=True, slots=True)
class _CalibratedGrid:
    """Level-zero XY grid of the selected original image."""

    height: int
    width: int
    microns_per_pixel: float


def image_kind(path: str | Path) -> ImageKind:
    """Return the decoder of a full-resolution image from its suffix."""
    suffix = Path(path).suffix.lower()
    if suffix not in _IMAGE_KINDS:
        message = (
            "fullres_image_file must be a TIFF/BigTIFF (.tif, .tiff, .btf), PNG or JPEG (.jpg, .jpeg) file; "
            f"found suffix {suffix!r}."
        )
        raise ValueError(message)
    return _IMAGE_KINDS[suffix]


def make_image_options(
    fullres_kind: ImageKind | None,
    *,
    data_axes: str | None,
    tiff_series: int | None,
    scale_factors: Sequence[int],
    chunks: int | tuple[int, int, int],
) -> ImageOptions:
    """Validate the image options against the kind of full-resolution image before any I/O.

    The rank of ``data_axes`` and the range of ``tiff_series`` depend on the
    file and are checked when it is read.
    """
    axes = validate_data_axes(data_axes)
    validate_tiff_index(tiff_series, parameter="tiff_series", optional=True)
    raster = make_raster_options(scale_factors, chunks)

    if fullres_kind != "tiff" and (data_axes is not None or tiff_series is not None):
        message = "data_axes and tiff_series require a TIFF-family fullres_image_file."
        raise ValueError(message)
    if fullres_kind is None and raster.scale_factors:
        message = "scale_factors require fullres_image_file; the hires and lowres images stay single-scale."
        raise ValueError(message)

    return ImageOptions(raster=raster, data_axes=axes, tiff_series=tiff_series)


def _parse(
    image: Array,
    *,
    channels: list[str] | None,
    transformations: dict[str, BaseTransformation],
    raster: RasterOptions,
    scale_factors: list[int] | None = None,
) -> DataArray | DataTree:
    """Build the image model with explicit channel names and chunks."""
    return Image2DModel.parse(
        image,
        dims=_IMAGE_AXES,
        c_coords=channels,
        rgb=False,
        transformations=transformations,
        scale_factors=scale_factors,
        chunks=raster.image_chunks(),
    )


def _read_fullres(
    path: ArtifactFile,
    *,
    options: ImageOptions,
    frame: str,
    version: str | None,
) -> tuple[DataArray | DataTree, _CalibratedGrid | None]:
    """Read bounded metadata of the full-resolution image and build it without pixel tasks."""
    context = ReaderErrorContext("visium", "full-resolution image", version)
    tiff = None
    kind = image_kind(path)
    if kind == "tiff":
        metadata = inspect_tiff(path, context=context)
        series = select_tiff_series(metadata, options.tiff_series, context=context, option="tiff_series")
        raster = read_tiff(metadata, series=series, level=0, context=context)
        tiff = (metadata, series)
    elif kind == "png":
        raster = read_png(path, context=context)
    else:
        raster = read_jpeg(path, context=context)

    image, channels = normalize_cyx_image(raster, options.data_axes, tiff=tiff, context=context, path=path)
    options.raster.validate_image_pyramid(image, element="the full-resolution image")
    calibration = None
    transformations: dict[str, BaseTransformation] = {frame: Identity()}
    # Reinterpreting the spatial axes invalidates their association with the OME sizes.
    declared = tuple(axis.lower() for axis in raster.axes)
    axes = declared if options.data_axes is None else tuple(options.data_axes)
    if tiff is not None and all(axis in declared and axes.index(axis) == declared.index(axis) for axis in ("x", "y")):
        metadata, series = tiff
        x, y = ome_pixel_sizes_um(metadata, series=series, axes=("x", "y"), context=context)
        if x is not None and y is not None and isclose(x, y, rel_tol=PIXEL_SIZE_RTOL):
            height, width = (int(size) for size in image.shape[1:])
            if not isfinite(x * max(height, width)):
                raise ReaderFormatError(
                    context=context,
                    path=path,
                    found=f"{x} µm/pixel for a {width} by {height} pixel grid",
                    expected="a finite full-resolution extent in micrometres",
                )
            calibration = _CalibratedGrid(height=height, width=width, microns_per_pixel=x)
            transformations["global"] = isotropic_scale(x)

    parsed = _parse(
        image,
        channels=channels,
        transformations=transformations,
        raster=options.raster,
        scale_factors=options.raster.model_scale_factors,
    )
    return parsed, calibration


def read_images(
    layout: VisiumLayout,
    *,
    dataset_id: str,
    frames: VisiumFrames,
    options: ImageOptions,
) -> tuple[dict[str, DataArray | DataTree], float | None]:
    """Return the lazy image elements and the full-resolution pixel size, if calibrated."""
    images: dict[str, DataArray | DataTree] = {}
    version = layout.format_version
    calibration = None
    if layout.fullres_image is not None:
        images[f"{dataset_id}_full_image"], calibration = _read_fullres(
            layout.fullres_image,
            options=options,
            frame=frames.fullres,
            version=version,
        )

    # Downsampled-image coordinates map back to full resolution by inverse scale.
    for name, path, frame, scale in (
        ("hires", layout.hires_image, frames.hires, frames.hires_scale),
        ("lowres", layout.lowres_image, frames.lowres, frames.lowres_scale),
    ):
        if path is None:
            continue
        context = ReaderErrorContext("visium", f"{name} image", version)
        source = read_png(path, context=context)
        image, channels = normalize_cyx_image(source, None, tiff=None, context=context, path=path)
        transformations: dict[str, BaseTransformation] = {
            frame: Identity(),
            frames.fullres: isotropic_scale(1 / scale),
        }
        if calibration is not None:
            height, width = (int(size) for size in image.shape[1:])
            expected_width, expected_height = calibration.width * scale, calibration.height * scale
            # The PNG has integer dimensions and Space Ranger serializes the scale with finite precision.
            if abs(width - expected_width) > 1 or abs(height - expected_height) > 1:
                raise ReaderFormatError(
                    context=context,
                    path=path,
                    found=f"{width} by {height} pixels",
                    expected=(
                        f"the full-resolution image grid scaled by {scale} ({expected_width:g} by {expected_height:g})"
                    ),
                )
            spacing = calibration.microns_per_pixel / scale
            if not isfinite(spacing * max(height, width)) or spacing <= 0:
                raise ReaderFormatError(
                    context=context,
                    path=path,
                    found=f"{spacing} µm/pixel for a {width} by {height} pixel grid",
                    expected="a positive finite scale and finite extent derived from the full-resolution image",
                )
            transformations["global"] = isotropic_scale(spacing)
        images[f"{dataset_id}_{name}_image"] = _parse(
            image, channels=channels, transformations=transformations, raster=options.raster
        )

    return images, None if calibration is None else calibration.microns_per_pixel
