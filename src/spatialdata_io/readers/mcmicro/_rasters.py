"""Construct lazy MCMICRO images and labels in their exact local pixel frames."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, cast

from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import inspect_tiff, normalize_raster_axes, read_tiff

if TYPE_CHECKING:
    import numpy as np
    from dask.array import Array
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.paths import ArtifactFile

_IMAGE_AXES = ("c", "y", "x")
_LABEL_AXES = ("y", "x")
_MIN_SCALE_FACTOR = 2
_MAX_EXACT_LABEL_BYTES = 4


@dataclass(frozen=True, slots=True)
class RasterOptions:
    """Own validated pyramid factors and canonical CYX chunk sizes."""

    scale_factors: tuple[int, ...] | None
    chunks: int | tuple[int, int, int] | None


def make_raster_options(
    scale_factors: Sequence[int] | None, chunks: int | tuple[int, int, int] | None
) -> RasterOptions:
    """Validate caller options once and own the borrowed factor sequence."""
    if scale_factors is not None and (
        isinstance(scale_factors, str | bytes) or not isinstance(scale_factors, Sequence)
    ):
        message = "scale_factors must be an ordered sequence of integer factors."
        raise TypeError(message)
    factors = None if scale_factors is None else tuple(scale_factors)
    if factors is not None and (
        not factors
        or any(isinstance(value, bool) or not isinstance(value, int) or value < _MIN_SCALE_FACTOR for value in factors)
    ):
        message = "scale_factors must be a nonempty sequence of integer downsampling factors greater than one."
        raise ValueError(message)

    if chunks is not None:
        sizes = chunks if isinstance(chunks, tuple) else (chunks,)
        if isinstance(chunks, tuple) and len(chunks) != len(_IMAGE_AXES):
            message = "chunks must contain three values in CYX order."
            raise ValueError(message)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in sizes):
            message = "chunks must be a positive integer or a tuple of three integers."
            raise TypeError(message)
        if any(value <= 0 for value in sizes):
            message = "chunks must contain only positive sizes."
            raise ValueError(message)
    return RasterOptions(factors, chunks)


def _read(path: ArtifactFile, artifact: str, target_axes: tuple[str, ...]) -> Array:
    """Inspect once and normalize the full-resolution series to named 2D axes."""
    context = ReaderErrorContext("mcmicro", artifact)
    try:
        metadata = inspect_tiff(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=context, path=path)
        raise
    if len(metadata.series) != 1:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"{len(metadata.series)} TIFF series",
            expected="one full-resolution series",
        )

    raster = read_tiff(metadata, series=0, level=0)
    try:
        return normalize_raster_axes(
            raster.data,
            raster.axes,
            target_axes=target_axes,
            add_missing=("c",) if target_axes == _IMAGE_AXES else (),
        )
    except ValueError as error:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=str(error),
            expected=f"spatial axes {target_axes} with only singleton additional axes",
        ) from error


def _check_pyramid(array: Array, options: RasterOptions) -> None:
    """Reject pyramid factors that would eliminate a spatial dimension."""
    if options.scale_factors is not None and min(array.shape[-2:]) < prod(options.scale_factors):
        message = f"Requested scale factors {options.scale_factors} exceed the spatial extent of an MCMICRO raster."
        raise ValueError(message)


def read_image(
    path: ArtifactFile, *, channels: tuple[str, ...], frame: str, options: RasterOptions
) -> tuple[DataArray | DataTree, tuple[int, int]]:
    """Return a lazy CYX image whose channels are the canonical marker panel."""
    image = _read(path, "image", _IMAGE_AXES)
    if image.shape[0] != len(channels):
        raise ReaderFormatError(
            context=ReaderErrorContext("mcmicro", "image channels"),
            path=path,
            found=f"{image.shape[0]} channels for {len(channels)} markers",
            expected="one registered channel per markers.csv row",
        )
    _check_pyramid(image, options)

    # TIFF inspection provides finite integer dimensions; Dask's general Array
    # annotation also permits unknown NaN dimensions.
    grid = cast("tuple[int, int]", image.shape[-2:])
    element = Image2DModel.parse(
        image,
        dims=_IMAGE_AXES,
        c_coords=list(channels),
        rgb=False,
        chunks=options.chunks or tuple(max(parts) for parts in image.chunks),
        scale_factors=options.scale_factors,
        transformations={frame: Identity()},
    )
    return element, grid


def read_labels(
    path: ArtifactFile, *, grid: tuple[int, int] | None, frame: str, options: RasterOptions
) -> tuple[DataArray | DataTree, np.dtype[np.generic]]:
    """Return lazy integer YX labels, requiring the image grid when one is given."""
    labels = _read(path, "labels", _LABEL_AXES)
    context = ReaderErrorContext("mcmicro", "labels")
    if grid is not None and labels.shape != grid:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"YX shape {labels.shape} for image grid {grid}",
            expected="labels on the full-resolution grid of their source image",
        )
    if labels.dtype.kind not in {"i", "u"}:
        raise ReaderFormatError(
            context=context, path=path, found=str(labels.dtype), expected="integer labels with background zero"
        )
    if options.scale_factors is not None and labels.dtype.itemsize > _MAX_EXACT_LABEL_BYTES:
        message = "Generated label pyramids require <=32-bit labels; use scale_factors=None for exact 64-bit IDs."
        raise ValueError(message)
    _check_pyramid(labels, options)

    chunks = options.chunks
    label_chunks = chunks[-2:] if isinstance(chunks, tuple) else chunks
    element = Labels2DModel.parse(
        labels,
        dims=_LABEL_AXES,
        chunks=label_chunks or tuple(max(parts) for parts in labels.chunks),
        scale_factors=options.scale_factors,
        transformations={frame: Identity()},
    )
    return element, labels.dtype
