"""Read the Visium HD microscope and CytAssist images lazily as CYX image elements.

Space Ranger writes two downsampled copies of the registered microscope image
and the CytAssist capture image; the full-resolution microscope image is an
input to Space Ranger that 10x sample downloads include beside the outputs::

    spatial/tissue_hires_image.png    full-resolution x tissue_hires_scalef (longest side 6000)
    spatial/tissue_lowres_image.png   full-resolution x tissue_lowres_scalef
    spatial/cytassist_image.tiff      CytAssist camera image (only with include_cytassist_image)
    tissue_image.tif                  full-resolution microscope image: TIFF, OME-TIFF,
                                      BigTIFF/BTF, PNG or JPEG

The ``spatial/`` directory is the root one or ``binned_outputs/square_002um/spatial``.
A full-resolution file is a TIFF when its first four bytes are a (Big)TIFF
signature, whatever its suffix (public ``.tif`` files can be BigTIFF);
otherwise ``.png``, ``.jpg`` and ``.jpeg`` select those decoders.

Axes and channels::

    source axes    TIFF: file-declared (``YX``, ``CYX``, ``YXS``, ...) or
                   ``data_axes`` (full-resolution only); PNG and JPEG: ``YXS``
    samples (S)    interleaved samples become channels
    other axes     singleton dimensions are dropped; any other extra axis fails
    channels       ``r, g, b(, a)`` for samples the file declares RGB(A);
                   otherwise OME channel names when all are nonempty and
                   unique, else OME channel IDs; positional integers without
                   either declaration

RGB(A) is declared by TIFF ``PhotometricInterpretation`` RGB (alpha through
``ExtraSamples``) or JPEG-compressed YCbCr, PNG colour types 2, 3 and 6, and
three-component JPEG. The channel count or dtype never makes an image RGB, so
a four-channel immunofluorescence TIFF keeps its OME names or positions. OME
names apply when the image's channel axis is the OME channel (or samples)
axis, or when the file has a single channel; a ``data_axes`` reinterpretation
of another axis as channels gets positions. Axes the file declares that cannot
become ``c, y, x`` are malformed content and raise ``ReaderFormatError``; axes
from ``data_axes`` that do not fit the file raise ``ValueError`` with a note
naming the image.

Coordinates: microscope images keep their pixels and get one affine into
``global`` micrometres, the least-squares fit of the exact image-to-slide map
(hires and lowres compose it with ``1 / tissue_*_scalef``); its maximum and
RMS residual over the capture grid are recorded in ``attrs``. The CytAssist
camera is registered projectively (``spot_colrow_to_cytassist_colrow`` in the
feature slice), so its image is resampled lazily with bilinear interpolation
onto an axis-aligned ``global`` grid at its own physical resolution and gets a
scale-and-translation transform.

A TIFF has one series unless ``tiff_series`` selects one of several; level 0 is
read. Only the full-resolution image gets generated pyramid levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
from dask import delayed
from scipy.ndimage import map_coordinates
from spatialdata.models import Image2DModel
from spatialdata.transformations import Affine

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    make_raster_options,
    normalize_cyx_image,
    read_jpeg,
    read_png,
    read_tiff,
    select_tiff_series,
    validate_data_axes,
    validate_tiff_index,
)
from spatialdata_io.readers.visium_hd._transforms import apply_projective

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.visium_hd._layout import ImageLayout
    from spatialdata_io.readers.visium_hd._metadata import ScaleFactors, SourceMetadata
    from spatialdata_io.readers.visium_hd._transforms import SlideFrame

_AXES = ("c", "y", "x")
_TIFF_SIGNATURES = frozenset((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"))


@dataclass(frozen=True, slots=True)
class ImageOptions:
    """Validated full-resolution selection and the shared pyramid and chunk options."""

    raster: RasterOptions
    data_axes: str | None
    tiff_series: int | None


@dataclass(frozen=True, slots=True)
class _Image:
    """One lazy CYX raster and its channel coordinates (``None`` means positional)."""

    pixels: da.Array
    channels: list[str] | None


def make_image_options(
    *,
    data_axes: str | None,
    tiff_series: int | None,
    scale_factors: Sequence[int],
    chunks: int | tuple[int, int, int],
) -> ImageOptions:
    """Validate the public raster options, in signature order, without opening images.

    The rank of ``data_axes`` and the range of ``tiff_series`` depend on the
    file and are checked when it is read.
    """
    axes = validate_data_axes(data_axes)
    validate_tiff_index(tiff_series, parameter="tiff_series", optional=True)
    raster = make_raster_options(scale_factors, chunks)
    return ImageOptions(raster, axes, tiff_series)


def _is_tiff(path: ArtifactFile, *, context: ReaderErrorContext) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) in _TIFF_SIGNATURES
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise


def _read_fullres(path: ArtifactFile, options: ImageOptions, *, version: str | None) -> _Image:
    """Read bounded metadata of the full-resolution image and build it without pixel tasks."""
    context = ReaderErrorContext("visium_hd", "full-resolution image", version)
    if _is_tiff(path, context=context):
        metadata = inspect_tiff(path, context=context)
        series = select_tiff_series(metadata, options.tiff_series, context=context, option="tiff_series")
        raster = read_tiff(metadata, series=series, level=0, context=context)
        image = _Image(
            *normalize_cyx_image(raster, options.data_axes, tiff=(metadata, series), context=context, path=path)
        )
    else:
        if options.data_axes is not None or options.tiff_series is not None:
            message = "data_axes and tiff_series apply only to TIFF-family full-resolution images."
            raise ValueError(message)

        suffix = path.suffix.lower()
        if suffix == ".png":
            raster = read_png(path, context=context)
        elif suffix in {".jpg", ".jpeg"}:
            raster = read_jpeg(path, context=context)
        else:
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"no TIFF signature and suffix {suffix!r}",
                expected="TIFF-family content, or a .png, .jpg or .jpeg file",
            )
        image = _Image(*normalize_cyx_image(raster, None, tiff=None, context=context, path=path))

    options.raster.validate_image_pyramid(image.pixels, element="the full-resolution image")
    return image


def _parse(
    image: _Image,
    matrix: NDArray[np.float64],
    options: RasterOptions,
    *,
    pyramid: bool = False,
) -> DataArray | DataTree:
    return Image2DModel.parse(
        image.pixels,
        dims=_AXES,
        c_coords=image.channels,
        rgb=False,
        transformations={"global": Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))},
        chunks=options.image_chunks(),
        scale_factors=options.model_scale_factors if pyramid else None,
    )


def _registration_attrs(frame: SlideFrame, scales: ScaleFactors) -> dict[str, float]:
    return {
        "registration_max_residual_um": frame.max_residual_um,
        "registration_rms_residual_um": frame.rms_residual_um,
        "space_ranger_microns_per_pixel": scales.microns_per_pixel,
    }


def read_images(  # noqa: PLR0913, RUF100 - Explicit layout, registration, calibration, options and version.
    layout: ImageLayout,
    frame: SlideFrame,
    scales: ScaleFactors,
    source: SourceMetadata | None,
    options: ImageOptions,
    *,
    version: str | None,
) -> dict[str, DataArray | DataTree]:
    """Construct selected rasters without executing pixel tasks.

    Microscopy images carry the measured slide affine; their residual attrs
    apply over the capture area. CytAssist is resampled exactly with bilinear
    interpolation at its physical resolution, preserving the source dtype.
    """
    images: dict[str, DataArray | DataTree] = {}
    hires_shape: tuple[int, int] | None = None
    for name, artifact, path, scale in (
        ("hires_image", "hires image", layout.hires, scales.hires),
        ("lowres_image", "lowres image", layout.lowres, scales.lowres),
    ):
        if path is None:
            continue

        context = ReaderErrorContext("visium_hd", artifact, version)
        image = _Image(
            *normalize_cyx_image(read_png(path, context=context), None, tiff=None, context=context, path=path)
        )
        if name == "hires_image":
            hires_shape = (int(image.pixels.shape[1]), int(image.pixels.shape[2]))
        matrix = frame.image_pixels_to_global @ np.diag([1 / scale, 1 / scale, 1.0])
        images[name] = _parse(image, matrix, options.raster)

    if layout.fullres is not None:
        image = _read_fullres(layout.fullres, options, version=version)
        extent = image.pixels.shape[1:]
        if hires_shape is not None and round(max(extent) * scales.hires) != max(hires_shape):
            raise ReaderFormatError(
                context=ReaderErrorContext("visium_hd", "full-resolution image", version),
                path=layout.fullres,
                found=f"shape {extent}, hires shape {hires_shape}, scale {scales.hires}",
                expected="matching longest sides after the declared hires scale and rounding",
            )
        images["full_image"] = _parse(image, frame.image_pixels_to_global, options.raster, pyramid=True)
    elif options.data_axes is not None or options.tiff_series is not None or options.raster.scale_factors:
        message = "data_axes, tiff_series and scale_factors require a full-resolution image."
        raise ValueError(message)

    for image in images.values():
        image.attrs.update(_registration_attrs(frame, scales))

    if layout.cytassist is not None:
        images["cytassist_image"] = _cytassist_image(layout.cytassist, frame, source, options.raster, version=version)

    return images


def _cytassist_image(
    path: ArtifactFile,
    frame: SlideFrame,
    source: SourceMetadata | None,
    options: RasterOptions,
    *,
    version: str | None,
) -> DataArray | DataTree:
    registration = ReaderErrorContext("visium_hd", "CytAssist registration", version)
    if source is None or source.cytassist is None:
        raise ReaderFormatError(
            context=registration,
            path=path,
            found="null CytAssist transform matrices",
            expected="a non-null CytAssist matrix pair for requested image placement",
        )

    context = ReaderErrorContext("visium_hd", "CytAssist image", version)
    metadata = inspect_tiff(path, context=context)
    series = select_tiff_series(metadata, None, context=context)
    raster = read_tiff(metadata, series=series, level=0, context=context)
    pixels, channels = normalize_cyx_image(raster, None, tiff=(metadata, series), context=context, path=path)
    warped, matrix = warp_cytassist(pixels, frame, source.cytassist, options, context=registration)
    parsed = _parse(_Image(warped, channels), matrix, options)
    parsed.attrs["interpolation"] = "bilinear"
    return parsed


def _warp_grid(
    source_shape: tuple[int, int], source_to_global: NDArray[np.float64], *, context: ReaderErrorContext
) -> tuple[tuple[int, int], NDArray[np.float64]]:
    height, width = source_shape
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    denominators = corners @ source_to_global[2, :2] + source_to_global[2, 2]
    if np.min(denominators) <= 0 <= np.max(denominators):
        raise ReaderFormatError(
            context=context,
            found="projective horizon intersects the image",
            expected="a finite image footprint in slide micrometres",
        )

    footprint = apply_projective(source_to_global, corners, context=context)
    # The Jacobian determinant gives local pixel area; its square root sets isotropic pitch.
    center = np.array([[width / 2, height / 2]], dtype=np.float64)
    denominator = float(center[0] @ source_to_global[2, :2] + source_to_global[2, 2])
    mapped = apply_projective(source_to_global, center, context=context)[0]
    jacobian = (source_to_global[:2, :2] - np.outer(mapped, source_to_global[2, :2])) / denominator
    step = float(np.sqrt(abs(np.linalg.det(jacobian))))

    origin = footprint.min(axis=0)
    dimensions = np.ceil((footprint.max(axis=0) - origin) / step).astype(np.int64)
    matrix = np.array([[step, 0, origin[0]], [0, step, origin[1]], [0, 0, 1]], dtype=np.float64)
    return (int(dimensions[1]), int(dimensions[0])), matrix


def _warp_tile(
    source: NDArray[np.generic],
    output_to_source: NDArray[np.float64],
    output_slice: tuple[int, int, int, int],
    source_origin: tuple[int, int],
    context: ReaderErrorContext,
) -> NDArray[np.generic]:
    y0, y1, x0, x1 = output_slice
    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=np.float64)

    # SpatialData transforms operate on pixel corners; scipy indexes centres at integers.
    points = np.column_stack(((xx + x0 + 0.5).ravel(), (yy + y0 + 0.5).ravel()))
    coordinates = apply_projective(output_to_source, points, context=context)
    coordinates -= np.array([source_origin[1] + 0.5, source_origin[0] + 0.5], dtype=np.float64)
    scipy_coordinates = np.stack((coordinates[:, 1], coordinates[:, 0])).reshape(2, y1 - y0, x1 - x0)

    return np.stack(
        [
            map_coordinates(channel, scipy_coordinates, order=1, mode="constant", cval=0, prefilter=False)
            for channel in source
        ]
    )


def warp_cytassist(
    pixels: da.Array,
    frame: SlideFrame,
    spot_to_cytassist: NDArray[np.float64],
    options: RasterOptions,
    *,
    context: ReaderErrorContext,
) -> tuple[da.Array, NDArray[np.float64]]:
    """Build exact projective warp tasks, each depending only on its source ROI.

    Output pixels form a global-axis-aligned grid covering the complete source
    footprint; pitch is the area-equivalent physical pixel size at its centre.
    Source TIFF tasks remain shared raster tasks and open their own resources.
    Both source and output pixel centres use the SpatialData corner-coordinate
    convention; the half-pixel conversions occur only at scipy's array boundary.
    ``context`` names the CytAssist registration in errors.
    """
    spot_to_grid_um = np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]], dtype=np.float64)
    source_to_global = frame.grid_um_to_global @ spot_to_grid_um @ np.linalg.inv(spot_to_cytassist)
    shape, matrix = _warp_grid((int(pixels.shape[1]), int(pixels.shape[2])), source_to_global, context=context)
    output_to_source = np.linalg.inv(source_to_global) @ matrix

    chunks = options.image_chunks()
    rows: list[da.Array] = []
    for y0 in range(0, shape[0], chunks["y"]):
        columns: list[da.Array] = []
        y1 = min(y0 + chunks["y"], shape[0])
        for x0 in range(0, shape[1], chunks["x"]):
            x1 = min(x0 + chunks["x"], shape[1])

            corners = np.array([[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]], dtype=np.float64)
            mapped = apply_projective(output_to_source, corners + 0.5, context=context) - 0.5
            # Include both interpolation neighbours, clipped to the available source pixels.
            low = np.maximum(np.floor(mapped.min(axis=0)).astype(np.int64), 0)
            high = np.minimum(np.ceil(mapped.max(axis=0)).astype(np.int64) + 1, [pixels.shape[2], pixels.shape[1]])

            tile_shape = (pixels.shape[0], y1 - y0, x1 - x0)
            if np.any(high <= low):
                tile = da.zeros(tile_shape, dtype=pixels.dtype, chunks=tile_shape)
            else:
                roi = pixels[:, low[1] : high[1], low[0] : high[0]]
                task = delayed(_warp_tile)(roi, output_to_source, (y0, y1, x0, x1), (int(low[1]), int(low[0])), context)
                tile = da.from_delayed(task, shape=tile_shape, dtype=pixels.dtype)
            columns.append(tile)
        rows.append(da.concatenate(columns, axis=2))

    return da.concatenate(rows, axis=1), matrix
