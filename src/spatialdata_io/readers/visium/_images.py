"""Lazy Visium raster loading and image-model construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.image import (
    LazyRaster,
    inspect_tiff,
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

# sphinx-autodoc-typehints currently renders PEP 695 aliases as ``TypeAliasType``.
ChunkValue: TypeAlias = int | tuple[int, ...] | None  # noqa: UP040
Chunks: TypeAlias = (  # noqa: UP040
    int | tuple[ChunkValue, ChunkValue, ChunkValue] | Mapping[str, ChunkValue]
)

DEFAULT_SPATIAL_CHUNK_SIZE = 1_024
TIFF_SUFFIXES = frozenset({".btf", ".tif", ".tiff"})
PNG_SUFFIXES = frozenset({".png"})
JPEG_SUFFIXES = frozenset({".jpeg", ".jpg"})
VALID_IMAGE_SUFFIXES = TIFF_SUFFIXES | PNG_SUFFIXES | JPEG_SUFFIXES
TARGET_AXES = ("c", "y", "x")
SPATIAL_RANK = 2
CHANNEL_AXES = frozenset({"c", "s"})
MAX_FULLRES_DOWNSCALING_STEPS = 4
FULLRES_DOWNSCALE_FACTOR = 2


@dataclass(frozen=True, slots=True)
class ImageOptions:
    """Validated immutable full-resolution selection and chunk options."""

    chunks: tuple[ChunkValue, ChunkValue, ChunkValue]
    data_axes: tuple[str, ...] | None
    tiff_series: int | None


def _validate_chunk_value(value: ChunkValue, *, axis: str) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        message = f"chunks for axis {axis!r} must not be a boolean."
        raise TypeError(message)
    if isinstance(value, int):
        if value <= 0:
            message = f"chunks for axis {axis!r} must be positive; found {value}."
            raise ValueError(message)
        return
    if not isinstance(value, tuple):
        message = f"chunks for axis {axis!r} must be an integer, tuple of integers, or None; found {value!r}."
        raise TypeError(message)
    if not value or any(isinstance(size, bool) or not isinstance(size, int) for size in value):
        message = f"chunks for axis {axis!r} must contain a nonempty tuple of integers; found {value!r}."
        raise TypeError(message)
    if any(size <= 0 for size in value):
        message = f"chunks for axis {axis!r} must contain only positive sizes; found {value!r}."
        raise ValueError(message)


def _normalize_chunks(chunks: Chunks | None) -> tuple[ChunkValue, ChunkValue, ChunkValue]:
    if chunks is None:
        return (None, DEFAULT_SPATIAL_CHUNK_SIZE, DEFAULT_SPATIAL_CHUNK_SIZE)
    if isinstance(chunks, bool):
        message = "chunks must be a chunk specification, not a boolean."
        raise TypeError(message)
    if isinstance(chunks, int):
        normalized: tuple[ChunkValue, ChunkValue, ChunkValue] = (chunks, chunks, chunks)
    elif isinstance(chunks, Mapping):
        if set(chunks) != set(TARGET_AXES):
            message = f"chunks keys must match canonical axes {TARGET_AXES}; found {tuple(chunks)}."
            raise ValueError(message)
        normalized = (chunks["c"], chunks["y"], chunks["x"])
    elif isinstance(chunks, tuple):
        if len(chunks) != len(TARGET_AXES):
            message = f"chunks length must match canonical axes {TARGET_AXES}; found {chunks!r}."
            raise ValueError(message)
        normalized = (chunks[0], chunks[1], chunks[2])
    else:
        message = f"chunks must be an integer, tuple, mapping, or None; found {type(chunks).__name__}."
        raise TypeError(message)
    for axis, value in zip(TARGET_AXES, normalized, strict=True):
        _validate_chunk_value(value, axis=axis)
    return normalized


def make_image_options(
    *,
    chunks: Chunks | None,
    data_axes: Sequence[str] | None,
    tiff_series: int | None,
) -> ImageOptions:
    """Validate caller-owned image options and return one immutable value."""
    if data_axes is None:
        normalized_axes = None
    else:
        normalized_axes = tuple(axis.lower() for axis in data_axes)
        if any(len(axis) != 1 or not axis.isalpha() for axis in normalized_axes):
            message = f"data_axes must contain single alphabetic axis names; found {tuple(data_axes)!r}."
            raise ValueError(message)
        if len(set(normalized_axes)) != len(normalized_axes):
            message = f"data_axes contains duplicate axes: {tuple(data_axes)!r}."
            raise ValueError(message)
    if tiff_series is not None and (
        isinstance(tiff_series, bool) or not isinstance(tiff_series, int) or tiff_series < 0
    ):
        message = f"tiff_series must be a non-negative integer or None; found {tiff_series!r}."
        raise ValueError(message)
    return ImageOptions(
        chunks=_normalize_chunks(chunks),
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
                    f"Full-resolution TIFF contains {len(metadata.series)} series; "
                    "select one explicitly with tiff_series."
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
    message = f"Unsupported image suffix {path.suffix!r}; expected one of {tuple(sorted(VALID_IMAGE_SUFFIXES))}."
    raise ValueError(message)


def _source_axes(raster: LazyRaster, *, override: tuple[str, ...] | None = None) -> tuple[str, ...]:
    axes = tuple(axis.lower() for axis in (raster.axes if override is None else override))
    if len(axes) != raster.data.ndim:
        message = f"Image axes rank {len(axes)} does not match raster shape {raster.data.shape}; found {axes!r}."
        raise ValueError(message)
    if len(set(axes)) != len(axes) or axes.count("x") != 1 or axes.count("y") != 1:
        message = f"Visium tissue images require unique X and Y axes; found {axes!r}."
        raise ValueError(message)

    channel_indices = [index for index, axis in enumerate(axes) if axis in CHANNEL_AXES]
    if len(channel_indices) > 1:
        message = f"Visium tissue images permit at most one channel or sample axis; found {axes!r}."
        raise ValueError(message)
    channel_index = channel_indices[0] if channel_indices else None
    for index, (axis, size) in enumerate(zip(axes, raster.data.shape, strict=True)):
        if axis in {"x", "y"} or index == channel_index or size == 1:
            continue
        message = (
            f"Visium tissue image axis {axis!r} has length {size} and no supported spatial or channel meaning; "
            "provide explicit data_axes for an ambiguous TIFF or select a spatially 2D series."
        )
        raise ValueError(message)
    return tuple("c" if index == channel_index else axis for index, axis in enumerate(axes))


def _fullres_scale_factors(shape: tuple[int, ...]) -> tuple[int, ...] | None:
    spatial_size = min(shape[-SPATIAL_RANK:])
    factors: list[int] = []
    while spatial_size >= FULLRES_DOWNSCALE_FACTOR and len(factors) < MAX_FULLRES_DOWNSCALING_STEPS:
        factors.append(FULLRES_DOWNSCALE_FACTOR)
        spatial_size //= FULLRES_DOWNSCALE_FACTOR
    return tuple(factors) or None


def _parse_image(
    raster: LazyRaster,
    *,
    source_axes: Sequence[str],
    transformations: dict[str, BaseTransformation],
    chunks: tuple[ChunkValue, ChunkValue, ChunkValue],
    scale_factors: tuple[int, ...] | None = None,
) -> DataArray | DataTree:
    normalized = normalize_raster_axes(
        raster.data,
        source_axes,
        target_axes=TARGET_AXES,
        add_missing=("c",) if "c" not in {axis.lower() for axis in source_axes} else (),
    )
    return Image2DModel.parse(
        normalized,
        dims=TARGET_AXES,
        transformations=transformations,
        scale_factors=scale_factors,
        chunks=dict(zip(TARGET_AXES, chunks, strict=True)),
        rgb=None,
    )


def read_images(
    layout: VisiumLayout,
    *,
    dataset_id: str,
    transforms: VisiumTransforms,
    options: ImageOptions,
) -> dict[str, DataArray | DataTree]:
    """Construct all selected image models without executing pixel tasks."""
    images: dict[str, DataArray | DataTree] = {}
    if layout.fullres_image is not None:
        raster = _read_raster(layout.fullres_image, options=options)
        source_axes = _source_axes(raster, override=options.data_axes)
        source_sizes = dict(zip(source_axes, raster.data.shape, strict=True))
        normalized_shape = tuple(int(source_sizes.get(axis, 1)) for axis in TARGET_AXES)
        images[f"{dataset_id}_full_image"] = _parse_image(
            raster,
            source_axes=source_axes,
            transformations={dataset_id: transforms.fullres},
            chunks=options.chunks,
            scale_factors=_fullres_scale_factors(normalized_shape),
        )
    elif options.data_axes is not None or options.tiff_series is not None:
        message = "data_axes and tiff_series require fullres_image_file."
        raise ValueError(message)
    if layout.hires_image is not None:
        raster = read_png(layout.hires_image)
        images[f"{dataset_id}_hires_image"] = _parse_image(
            raster,
            source_axes=_source_axes(raster),
            transformations={
                f"{dataset_id}_downscaled_hires": Identity(),
                dataset_id: transforms.hires.inverse(),
            },
            chunks=options.chunks,
        )
    if layout.lowres_image is not None:
        raster = read_png(layout.lowres_image)
        images[f"{dataset_id}_lowres_image"] = _parse_image(
            raster,
            source_axes=_source_axes(raster),
            transformations={
                f"{dataset_id}_downscaled_lowres": Identity(),
                dataset_id: transforms.lowres.inverse(),
            },
            chunks=options.chunks,
        )
    return images
