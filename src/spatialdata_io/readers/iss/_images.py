"""Interpret ISS TIFF axes and construct lazy image and label elements."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import inspect_tiff, normalize_raster_axes, read_tiff

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from dask.array import Array
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image.tiff import TiffMetadata
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.iss._layout import _IssLayout

_IMAGE_AXES = ("c", "y", "x")
_LABEL_AXES = ("y", "x")
_MAX_EXACT_LABEL_BYTES = 4
_MIN_SCALE_FACTOR = 2


@dataclass(frozen=True, slots=True)
class _ImageOptions:
    """Own validated axis, pyramid and chunk choices independently of caller lists."""

    data_axes: str | None
    image_scale_factors: tuple[int, ...] | None
    labels_scale_factors: tuple[int, ...] | None
    chunks: int | tuple[int, int, int] | None


@dataclass(frozen=True, slots=True)
class _IssRasters:
    """Carry lazy linked-frame rasters and the validated integer label dtype."""

    image: DataArray | DataTree
    labels: DataArray | DataTree
    label_dtype: np.dtype[np.generic]


def _image_options(
    *,
    data_axes: str | None,
    image_scale_factors: Sequence[int] | None,
    labels_scale_factors: Sequence[int] | None,
    chunks: int | tuple[int, int, int] | None,
) -> _ImageOptions:
    """Validate public raster options and materialize borrowed factor sequences."""
    if data_axes is not None:
        if not isinstance(data_axes, str):
            message = "data_axes must be a string or None."
            raise TypeError(message)
        if (
            not data_axes
            or not data_axes.isascii()
            or not data_axes.isalpha()
            or len(set(data_axes.lower())) != len(data_axes)
            or not {"x", "y"}.issubset(data_axes.lower())
        ):
            message = "data_axes must contain unique alphabetic axis names including X and Y."
            raise ValueError(message)
        data_axes = data_axes.lower()

    # Own borrowed sequences before lazy products retain the selected configuration.
    factors: list[tuple[int, ...] | None] = []
    for name, supplied in (
        ("image_scale_factors", image_scale_factors),
        ("labels_scale_factors", labels_scale_factors),
    ):
        if supplied is None:
            factors.append(None)
            continue
        values = tuple(supplied)
        if not values or any(
            isinstance(value, bool) or not isinstance(value, int) or value < _MIN_SCALE_FACTOR for value in values
        ):
            message = f"{name} must be a nonempty sequence of integer downsampling factors greater than one."
            raise ValueError(message)
        factors.append(values)
    _validate_chunks(chunks)
    return _ImageOptions(data_axes, factors[0], factors[1], chunks)


def _validate_chunks(chunks: int | tuple[int, int, int] | None) -> None:
    """Require positive model chunk sizes in canonical CYX order."""
    if chunks is None:
        return
    sizes = chunks if isinstance(chunks, tuple) else (chunks,)
    if isinstance(chunks, tuple) and len(chunks) != len(_IMAGE_AXES):
        message = "chunks must contain three values in CYX order."
        raise ValueError(message)
    if any(isinstance(size, bool) or not isinstance(size, int) for size in sizes):
        message = "chunks must be a positive integer or a tuple of three integers."
        raise TypeError(message)
    if any(size <= 0 for size in sizes):
        message = "chunks must contain only positive sizes."
        raise ValueError(message)


def _read_raster(
    path: ArtifactFile, *, name: str, target_axes: tuple[str, ...], data_axes: str | None = None
) -> tuple[Array, TiffMetadata]:
    """Inspect one full-resolution series and normalize named axes lazily."""
    context = ReaderErrorContext("iss", name)
    try:
        metadata = inspect_tiff(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=context, path=path)
        raise
    if len(metadata.series) != 1:
        raise ReaderFormatError(
            context=context, path=path, found=f"{len(metadata.series)} series", expected="exactly one TIFF series"
        )
    raster = read_tiff(metadata, series=0, level=0)
    axes = tuple(axis.lower() for axis in (raster.axes if data_axes is None else data_axes))
    # C is scientific channels. S only loses its axis when it is singleton;
    # interleaved samples need an explicit raw-image axis interpretation.
    try:
        array = normalize_raster_axes(
            raster.data, axes, target_axes=target_axes, add_missing=("c",) if "c" in target_axes else ()
        )
    except ValueError as error:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=str(error),
            expected=f"unambiguous 2D axes {target_axes}; singleton nonspatial axes only",
        ) from error
    return array, metadata


def _read_rasters(layout: _IssLayout, options: _ImageOptions) -> _IssRasters:
    """Construct pixel-aligned raw and labels models without executing pixel tasks."""
    image, metadata = _read_raster(
        layout.raw_image, name="raw image", target_axes=_IMAGE_AXES, data_axes=options.data_axes
    )
    labels, _ = _read_raster(layout.labels, name="labels", target_axes=_LABEL_AXES)
    if image.shape[-2:] != labels.shape:
        raise ReaderFormatError(
            context=ReaderErrorContext("iss", "raster alignment"),
            path=layout.root,
            found=f"raw YX shape {image.shape[-2:]} and labels shape {labels.shape}",
            expected="the same full-resolution pixel extent for raw and labels",
        )
    if labels.dtype.kind not in {"i", "u"}:
        raise ReaderFormatError(
            context=ReaderErrorContext("iss", "labels"),
            path=layout.labels,
            found=f"dtype {labels.dtype}",
            expected="integer label pixels, with zero reserved for background",
        )
    if options.labels_scale_factors is not None and labels.dtype.itemsize > _MAX_EXACT_LABEL_BYTES:
        message = (
            "Generated label pyramids require <=32-bit integer labels; the upstream model rounds 64-bit IDs. "
            "Use labels_scale_factors=None for exact 64-bit labels."
        )
        raise ValueError(message)
    for array, factors, path in (
        (image, options.image_scale_factors, layout.raw_image),
        (labels, options.labels_scale_factors, layout.labels),
    ):
        if factors is not None and min(array.shape[-2:]) < prod(factors):
            message = f"Requested scale factors {factors} exceed the spatial extent of ISS raster {path}."
            raise ValueError(message)

    # OME names describe scientific channels, regardless of a three/four-channel count.
    channel_names = None
    ome = metadata.ome
    if ome is not None:
        channels = ome.images[0].pixels.channels
        if len(channels) != image.shape[0]:
            raise ReaderFormatError(
                context=ReaderErrorContext("iss", "raw image channels"),
                path=layout.raw_image,
                found=f"{len(channels)} OME channels and {image.shape[0]} raster channels",
                expected="one OME channel record per output channel",
            )
        names = [channel.name for channel in channels]
        channel_names = (
            [str(name) for name in names]
            if all(names) and len(set(names)) == len(names)
            else [channel.id for channel in channels]
        )
        if len(set(channel_names)) != len(channel_names):
            raise ReaderFormatError(
                context=ReaderErrorContext("iss", "raw image channels"),
                path=layout.raw_image,
                found=f"duplicate channel identifiers {channel_names!r}",
                expected="unique channel names or OME IDs",
            )

    # Integer chunk sizes work at every pyramid level; complete base partitions do not.
    image_chunks = options.chunks or tuple(max(parts) for parts in image.chunks)
    label_chunks = image_chunks[-2:] if isinstance(image_chunks, tuple) else image_chunks
    if options.chunks is None:
        label_chunks = tuple(max(parts) for parts in labels.chunks)
    return _IssRasters(
        Image2DModel.parse(
            image,
            dims=_IMAGE_AXES,
            c_coords=channel_names,
            rgb=False,
            chunks=image_chunks,
            scale_factors=options.image_scale_factors,
            transformations={"global": Identity()},
        ),
        Labels2DModel.parse(
            labels,
            dims=_LABEL_AXES,
            chunks=label_chunks,
            scale_factors=options.labels_scale_factors,
            transformations={"global": Identity()},
        ),
        labels.dtype,
    )
