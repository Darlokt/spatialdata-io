"""Lazy Visium raster loading and image-model construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity

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

if TYPE_CHECKING:
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.visium._geometry import VisiumTransforms
    from spatialdata_io.readers.visium._layout import VisiumLayout

TIFF_SUFFIXES = frozenset({".btf", ".tif", ".tiff"})
PNG_SUFFIXES = frozenset({".png"})
JPEG_SUFFIXES = frozenset({".jpeg", ".jpg"})
VALID_IMAGE_SUFFIXES = TIFF_SUFFIXES | PNG_SUFFIXES | JPEG_SUFFIXES
TARGET_AXES = ("c", "y", "x")


@dataclass(frozen=True, slots=True)
class ImageOptions:
    """Validated immutable full-resolution selection, pyramid, and chunk options."""

    raster: RasterOptions
    data_axes: tuple[str, ...] | None
    tiff_series: int | None


def _chunk_sizes(chunks: int | tuple[int, int, int] | Mapping[str, int]) -> int | tuple[int, int, int]:
    if not isinstance(chunks, Mapping):
        return chunks
    if set(chunks) != set(TARGET_AXES):
        message = f"chunks keys must match canonical axes {TARGET_AXES}. Found {tuple(chunks)}."
        raise ValueError(message)
    return (chunks["c"], chunks["y"], chunks["x"])


def make_image_options(
    *,
    scale_factors: Sequence[int],
    chunks: int | tuple[int, int, int] | Mapping[str, int],
    data_axes: Sequence[str] | None,
    tiff_series: int | None,
) -> ImageOptions:
    """Validate caller-owned image options and return one immutable value."""
    if data_axes is None:
        normalized_axes = None
    else:
        normalized_axes = tuple(axis.lower() for axis in data_axes)
        if any(len(axis) != 1 or not axis.isalpha() for axis in normalized_axes):
            message = f"data_axes must contain single alphabetic axis names. Found {tuple(data_axes)!r}."
            raise ValueError(message)
        if len(set(normalized_axes)) != len(normalized_axes):
            message = f"data_axes contains duplicate axes: {tuple(data_axes)!r}."
            raise ValueError(message)

    if tiff_series is not None and (
        isinstance(tiff_series, bool) or not isinstance(tiff_series, int) or tiff_series < 0
    ):
        message = f"tiff_series must be a non-negative integer or None. Found {tiff_series!r}."
        raise ValueError(message)

    # Mapping values reach the shared validation, which rejects None and irregular partitions.
    return ImageOptions(
        raster=make_raster_options(scale_factors, _chunk_sizes(chunks)),
        data_axes=normalized_axes,
        tiff_series=tiff_series,
    )


def _read_raster(path: ArtifactFile, *, options: ImageOptions) -> LazyRaster:
    suffix = path.suffix.lower()
    if suffix in TIFF_SUFFIXES:
        metadata = inspect_tiff(path)
        series = options.tiff_series
        if series is None:
            if len(metadata.series) != 1:
                message = (
                    f"Full-resolution TIFF contains {len(metadata.series)} series. "
                    "Select one explicitly with tiff_series."
                )
                raise ValueError(message)
            series = 0
        return read_tiff(metadata, series=series, level=0)

    if options.tiff_series is not None:
        message = "tiff_series can only select TIFF-family full-resolution images."
        raise ValueError(message)
    if options.data_axes is not None:
        message = "data_axes can only override TIFF-family full-resolution image axes."
        raise ValueError(message)
    if suffix in PNG_SUFFIXES:
        return read_png(path)
    if suffix in JPEG_SUFFIXES:
        return read_jpeg(path)

    message = f"Unsupported image suffix {path.suffix!r}. Expected one of {tuple(sorted(VALID_IMAGE_SUFFIXES))}."
    raise ValueError(message)


def _parse_image(
    raster: LazyRaster,
    *,
    source_axes: tuple[str, ...],
    transformations: dict[str, BaseTransformation],
    chunks: dict[str, int],
    scale_factors: list[int] | None = None,
) -> DataArray | DataTree:
    axes = tuple("c" if axis == "s" else axis for axis in map(str.lower, source_axes))
    normalized = normalize_raster_axes(
        raster.data,
        axes,
        target_axes=TARGET_AXES,
        add_missing=("c",),
    )

    return Image2DModel.parse(
        normalized,
        dims=TARGET_AXES,
        transformations=transformations,
        scale_factors=scale_factors,
        chunks=chunks,
        rgb=None,
    )


def read_images(
    layout: VisiumLayout,
    *,
    dataset_id: str,
    transforms: VisiumTransforms,
    options: ImageOptions,
) -> dict[str, DataArray | DataTree]:
    """Construct all selected image models without executing pixel tasks.

    Only the full-resolution image receives generated pyramid levels; the hires
    and lowres PNGs are vendor-downsampled images and stay single-scale.
    """
    images: dict[str, DataArray | DataTree] = {}
    chunks = options.raster.image_chunks()
    if layout.fullres_image is not None:
        raster = _read_raster(layout.fullres_image, options=options)
        images[f"{dataset_id}_full_image"] = _parse_image(
            raster,
            source_axes=raster.axes if options.data_axes is None else options.data_axes,
            transformations={dataset_id: transforms.fullres},
            chunks=chunks,
            scale_factors=options.raster.model_scale_factors,
        )
    elif options.data_axes is not None or options.tiff_series is not None or options.raster.scale_factors:
        message = "data_axes, tiff_series and scale_factors require fullres_image_file."
        raise ValueError(message)

    # Downsampled-image coordinates map back to full resolution by inverse scale.
    if layout.hires_image is not None:
        raster = read_png(layout.hires_image)
        images[f"{dataset_id}_hires_image"] = _parse_image(
            raster,
            source_axes=raster.axes,
            transformations={
                f"{dataset_id}_downscaled_hires": Identity(),
                dataset_id: transforms.hires.inverse(),
            },
            chunks=chunks,
        )

    if layout.lowres_image is not None:
        raster = read_png(layout.lowres_image)
        images[f"{dataset_id}_lowres_image"] = _parse_image(
            raster,
            source_axes=raster.axes,
            transformations={
                f"{dataset_id}_downscaled_lowres": Identity(),
                dataset_id: transforms.lowres.inverse(),
            },
            chunks=chunks,
        )

    return images
