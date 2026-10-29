"""Lazy microscopy rasters and bounded exact CytAssist reprojection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
from dask import delayed
from scipy.ndimage import map_coordinates
from spatialdata.models import Image2DModel
from spatialdata.transformations import Affine

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import (
    LazyRaster,
    RasterOptions,
    inspect_tiff,
    make_raster_options,
    normalize_raster_axes,
    read_jpeg,
    read_png,
    read_tiff,
)
from spatialdata_io.readers.visium_hd._transforms import apply_projective

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.visium_hd._layout import ImageLayout
    from spatialdata_io.readers.visium_hd._metadata import ScaleFactors, SourceMetadata
    from spatialdata_io.readers.visium_hd._transforms import SlideFrame

_AXES = ("c", "y", "x")
_TIFF_HEADERS = frozenset((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"))


@dataclass(frozen=True, slots=True)
class ImageOptions:
    """Owned validated full-resolution axes, TIFF selection and raster options."""

    raster: RasterOptions
    data_axes: tuple[str, ...] | None
    tiff_series: int | None


def make_image_options(
    *,
    scale_factors: Sequence[int],
    chunks: int | tuple[int, int, int],
    data_axes: Sequence[str] | None,
    tiff_series: int | None,
) -> ImageOptions:
    """Validate public raster options without opening images."""
    if data_axes is not None and (
        not isinstance(data_axes, Sequence) or any(not isinstance(axis, str) for axis in data_axes)
    ):
        message = "data_axes must be a sequence of single alphabetic axis names."
        raise TypeError(message)

    axes = None if data_axes is None else tuple(axis.lower() for axis in data_axes)
    if axes is not None and (
        not axes or any(len(axis) != 1 or not axis.isalpha() for axis in axes) or len(set(axes)) != len(axes)
    ):
        message = "data_axes must contain unique single alphabetic axis names."
        raise ValueError(message)

    if tiff_series is not None and (
        isinstance(tiff_series, bool) or not isinstance(tiff_series, int) or tiff_series < 0
    ):
        message = "tiff_series must be a non-negative integer or None."
        raise ValueError(message)

    return ImageOptions(make_raster_options(scale_factors, chunks), axes, tiff_series)


def _read_fullres(path: ArtifactFile, options: ImageOptions) -> LazyRaster:
    with path.open("rb") as handle:
        is_tiff = handle.read(4) in _TIFF_HEADERS

    if is_tiff:
        metadata = inspect_tiff(path)
        series = options.tiff_series
        if series is None:
            if len(metadata.series) != 1:
                message = f"Full-resolution TIFF has {len(metadata.series)} series; select tiff_series explicitly."
                raise ValueError(message)
            series = 0
        return read_tiff(metadata, series=series, level=0)

    if options.data_axes is not None or options.tiff_series is not None:
        message = "data_axes and tiff_series apply only to TIFF-family full-resolution images."
        raise ValueError(message)

    if path.suffix.lower() == ".png":
        return read_png(path)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return read_jpeg(path)

    message = f"Unsupported full-resolution image format: {path.name!r}."
    raise ValueError(message)


def _normalize(raster: LazyRaster, override: tuple[str, ...] | None = None) -> da.Array:
    axes = tuple(axis.lower() for axis in (raster.axes if override is None else override))
    channels = [index for index, axis in enumerate(axes) if axis in {"s", "c"}]
    if len(channels) > 1:
        message = f"Images permit at most one channel/sample axis; found {axes!r}."
        raise ValueError(message)

    renamed = tuple("c" if index in channels else axis for index, axis in enumerate(axes))
    return normalize_raster_axes(raster.data, renamed, target_axes=_AXES, add_missing=("c",) if not channels else ())


def _parse_image(
    pixels: da.Array,
    matrix: NDArray[np.float64],
    options: RasterOptions,
    *,
    pyramid: bool = False,
) -> DataArray | DataTree:
    return Image2DModel.parse(
        pixels,
        dims=_AXES,
        transformations={"global": Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))},
        chunks=options.image_chunks(),
        scale_factors=options.model_scale_factors if pyramid else None,
        rgb=None,
    )


def _registration_attrs(frame: SlideFrame, scales: ScaleFactors) -> dict[str, float]:
    return {
        "registration_max_residual_um": frame.max_residual_um,
        "registration_rms_residual_um": frame.rms_residual_um,
        "space_ranger_microns_per_pixel": scales.microns_per_pixel,
    }


def read_images(
    layout: ImageLayout,
    frame: SlideFrame,
    scales: ScaleFactors,
    source: SourceMetadata | None,
    options: ImageOptions,
) -> dict[str, DataArray | DataTree]:
    """Construct selected rasters without executing pixel tasks.

    Microscopy images carry the measured slide affine; their residual attrs
    apply over the capture area. CytAssist is resampled exactly with bilinear
    interpolation at its physical resolution, preserving the source dtype.
    """
    images: dict[str, DataArray | DataTree] = {}
    hires_shape: tuple[int, int] | None = None
    for name, path, scale in (
        ("hires_image", layout.hires, scales.hires),
        ("lowres_image", layout.lowres, scales.lowres),
    ):
        if path is None:
            continue

        pixels = _normalize(read_png(path))
        if name == "hires_image":
            hires_shape = (int(pixels.shape[1]), int(pixels.shape[2]))
        matrix = frame.image_pixels_to_global @ np.diag([1 / scale, 1 / scale, 1.0])
        images[name] = _parse_image(pixels, matrix, options.raster)

    if layout.fullres is not None:
        pixels = _normalize(_read_fullres(layout.fullres, options), options.data_axes)
        if hires_shape is not None and round(max(pixels.shape[1:]) * scales.hires) != max(hires_shape):
            raise ReaderFormatError(
                context=ReaderErrorContext("visium_hd", "full-resolution image"),
                path=layout.fullres,
                found=f"shape {pixels.shape[1:]}, hires shape {hires_shape}, scale {scales.hires}",
                expected="matching longest sides after the declared hires scale and rounding",
            )
        images["full_image"] = _parse_image(pixels, frame.image_pixels_to_global, options.raster, pyramid=True)
    elif options.data_axes is not None or options.tiff_series is not None or options.raster.scale_factors:
        message = "data_axes, tiff_series and scale_factors require a full-resolution image."
        raise ValueError(message)

    for image in images.values():
        image.attrs.update(_registration_attrs(frame, scales))

    if layout.cytassist is not None:
        images["cytassist_image"] = _cytassist_image(layout.cytassist, frame, source, options.raster)

    return images


def _cytassist_image(
    path: ArtifactFile, frame: SlideFrame, source: SourceMetadata | None, options: RasterOptions
) -> DataArray | DataTree:
    if source is None or source.spot_to_cytassist is None or source.cytassist_to_spot is None:
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "CytAssist registration"),
            path=path,
            found="null CytAssist transform matrices",
            expected="a non-null CytAssist matrix pair for requested image placement",
        )

    metadata = inspect_tiff(path)
    if len(metadata.series) != 1:
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "CytAssist image"),
            path=path,
            found=f"{len(metadata.series)} TIFF series",
            expected="one unambiguous CytAssist image series",
        )

    pixels = _normalize(read_tiff(metadata, series=0, level=0))
    warped, matrix = warp_cytassist(pixels, frame, source.spot_to_cytassist, options)
    image = _parse_image(warped, matrix, options)
    image.attrs["interpolation"] = "bilinear"
    return image


def _warp_grid(
    source_shape: tuple[int, int], source_to_global: NDArray[np.float64]
) -> tuple[tuple[int, int], NDArray[np.float64]]:
    height, width = source_shape
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    denominators = corners @ source_to_global[2, :2] + source_to_global[2, 2]
    if np.min(denominators) <= 0 <= np.max(denominators):
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "CytAssist registration"),
            found="projective horizon intersects the image",
            expected="a finite image footprint in slide micrometres",
        )

    footprint = apply_projective(source_to_global, corners)
    # The Jacobian determinant gives local pixel area; its square root sets isotropic pitch.
    center = np.array([[width / 2, height / 2]], dtype=np.float64)
    denominator = float(center[0] @ source_to_global[2, :2] + source_to_global[2, 2])
    mapped = apply_projective(source_to_global, center)[0]
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
) -> NDArray[np.generic]:
    y0, y1, x0, x1 = output_slice
    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=np.float64)

    # SpatialData transforms operate on pixel corners; scipy indexes centres at integers.
    points = np.column_stack(((xx + x0 + 0.5).ravel(), (yy + y0 + 0.5).ravel()))
    coordinates = apply_projective(output_to_source, points)
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
) -> tuple[da.Array, NDArray[np.float64]]:
    """Build exact projective warp tasks, each depending only on its source ROI.

    Output pixels form a global-axis-aligned grid covering the complete source
    footprint; pitch is the area-equivalent physical pixel size at its centre.
    Source TIFF tasks remain shared raster tasks and open their own resources.
    Both source and output pixel centres use the SpatialData corner-coordinate
    convention; the half-pixel conversions occur only at scipy's array boundary.
    """
    spot_to_grid_um = np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]], dtype=np.float64)
    source_to_global = frame.grid_um_to_global @ spot_to_grid_um @ np.linalg.inv(spot_to_cytassist)
    shape, matrix = _warp_grid((int(pixels.shape[1]), int(pixels.shape[2])), source_to_global)
    output_to_source = np.linalg.inv(source_to_global) @ matrix

    chunks = options.image_chunks()
    rows: list[da.Array] = []
    for y0 in range(0, shape[0], chunks["y"]):
        columns: list[da.Array] = []
        y1 = min(y0 + chunks["y"], shape[0])
        for x0 in range(0, shape[1], chunks["x"]):
            x1 = min(x0 + chunks["x"], shape[1])

            corners = np.array([[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]], dtype=np.float64)
            mapped = apply_projective(output_to_source, corners + 0.5) - 0.5
            # Include both interpolation neighbours, clipped to the available source pixels.
            low = np.maximum(np.floor(mapped.min(axis=0)).astype(np.int64), 0)
            high = np.minimum(np.ceil(mapped.max(axis=0)).astype(np.int64) + 1, [pixels.shape[2], pixels.shape[1]])

            tile_shape = (pixels.shape[0], y1 - y0, x1 - x0)
            if np.any(high <= low):
                tile = da.zeros(tile_shape, dtype=pixels.dtype, chunks=tile_shape)
            else:
                roi = pixels[:, low[1] : high[1], low[0] : high[0]]
                task = delayed(_warp_tile)(roi, output_to_source, (y0, y1, x0, x1), (int(low[1]), int(low[0])))
                tile = da.from_delayed(task, shape=tile_shape, dtype=pixels.dtype)
            columns.append(tile)
        rows.append(da.concatenate(columns, axis=2))

    return da.concatenate(rows, axis=1), matrix
