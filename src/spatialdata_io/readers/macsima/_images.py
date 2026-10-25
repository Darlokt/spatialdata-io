"""Construct lazy, padded MACSima channel stacks as SpatialData images."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import dask.array as da
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity, Scale

from spatialdata_io.readers._utils.image import normalize_raster_axes, read_tiff
from spatialdata_io.readers.macsima._channels import NO_PADDING

if TYPE_CHECKING:
    from xarray import DataArray, DataTree

    from spatialdata_io.readers.macsima._channels import PlannedChannel

_IMAGE_AXES = ("c", "y", "x")
_MIN_SCALE_FACTOR = 2


@dataclass(frozen=True, slots=True)
class RasterOptions:
    """Validated pyramid factors, CYX chunks, and the physical-frame choice."""

    scale_factors: tuple[int, ...] | None
    chunks: int | tuple[int, int, int]
    include_transformations: bool


def make_raster_options(
    scale_factors: Sequence[int] | None, chunks: int | tuple[int, int, int], *, include_transformations: bool
) -> RasterOptions:
    """Validate caller options once and own the borrowed factor sequence."""
    if scale_factors is not None and (
        isinstance(scale_factors, str | bytes) or not isinstance(scale_factors, Sequence)
    ):
        message = "scale_factors must be an ordered sequence of integer factors."
        raise TypeError(message)
    factors = None if scale_factors is None else tuple(scale_factors)
    if factors is not None and any(isinstance(value, bool) or not isinstance(value, int) for value in factors):
        message = "scale_factors must contain only integers."
        raise TypeError(message)
    if factors is not None and (not factors or any(value < _MIN_SCALE_FACTOR for value in factors)):
        message = "scale_factors must be a nonempty sequence of downsampling factors greater than one."
        raise ValueError(message)

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
    if not isinstance(include_transformations, bool):
        message = "include_transformations must be a boolean."
        raise TypeError(message)

    return RasterOptions(factors, chunks, include_transformations)


def _channel_array(channel: PlannedChannel) -> da.Array:
    """Read one channel lazily and zero-pad it onto the stack canvas."""
    raster = read_tiff(channel.source.tiff, series=0, level=0)
    plane = normalize_raster_axes(raster.data, raster.axes, target_axes=("y", "x"))
    if channel.padding == NO_PADDING:
        return plane
    return da.pad(plane, channel.padding, mode="constant", constant_values=0)


def make_image(
    channels: Sequence[PlannedChannel], *, frame: str, pixel_size_um: float, options: RasterOptions
) -> DataArray | DataTree:
    """Stack planned channels as a lazy CYX image in the stack's coordinate system.

    With transformations, ``frame`` is in micrometers through an isotropic
    scale; otherwise it is the pixel grid itself. Building the image reads no
    pixels; channels shared with another stack reuse the same graph keys.
    """
    stack = da.stack([_channel_array(channel) for channel in channels])
    if options.scale_factors is not None and min(stack.shape[-2:]) < prod(options.scale_factors):
        message = (
            f"Requested scale factors {options.scale_factors} exceed the spatial extent of MACSima stack {frame!r}."
        )
        raise ValueError(message)

    transformation = (
        Scale([pixel_size_um, pixel_size_um], axes=("x", "y")) if options.include_transformations else Identity()
    )
    return Image2DModel.parse(
        stack,
        dims=_IMAGE_AXES,
        c_coords=[channel.name for channel in channels],
        rgb=False,
        chunks=options.chunks,
        scale_factors=options.scale_factors,
        transformations={frame: transformation},
    )
