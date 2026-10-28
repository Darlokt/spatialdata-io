"""Construct lazy, padded MACSima channel stacks as SpatialData images."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import dask.array as da
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity, Scale

from spatialdata_io.readers._utils.image import normalize_raster_axes, read_tiff
from spatialdata_io.readers.macsima._channels import NO_PADDING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers.macsima._channels import PlannedChannel

_IMAGE_AXES = ("c", "y", "x")


def _channel_array(channel: PlannedChannel) -> da.Array:
    """Read one channel lazily and zero-pad it onto the stack canvas."""
    raster = read_tiff(channel.source.tiff, series=0, level=0)
    plane = normalize_raster_axes(raster.data, raster.axes, target_axes=("y", "x"))
    if channel.padding == NO_PADDING:
        return plane
    return da.pad(plane, channel.padding, mode="constant", constant_values=0)


def make_image(
    channels: Sequence[PlannedChannel],
    *,
    frame: str,
    pixel_size_um: float,
    include_transformations: bool,
    raster: RasterOptions,
) -> DataArray | DataTree:
    """Stack planned channels as a lazy CYX image in the stack's coordinate system.

    With transformations, ``frame`` is in micrometers through an isotropic
    scale; otherwise it is the pixel grid itself. Building the image reads no
    pixels; channels shared with another stack reuse the same graph keys.
    """
    stack = da.stack([_channel_array(channel) for channel in channels])
    if raster.scale_factors and min(stack.shape[-2:]) < prod(raster.scale_factors):
        message = (
            f"Requested scale factors {raster.scale_factors} exceed the spatial extent of MACSima stack {frame!r}."
        )
        raise ValueError(message)

    transformation = Scale([pixel_size_um, pixel_size_um], axes=("x", "y")) if include_transformations else Identity()
    return Image2DModel.parse(
        stack,
        dims=_IMAGE_AXES,
        c_coords=[channel.name for channel in channels],
        rgb=False,
        chunks=raster.image_chunks(),
        scale_factors=raster.model_scale_factors,
        transformations={frame: transformation},
    )
