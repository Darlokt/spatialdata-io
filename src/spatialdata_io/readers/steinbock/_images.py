"""Interpret Steinbock OME channels and construct lazy per-sample rasters."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import inspect_tiff, normalize_raster_axes, read_tiff

if TYPE_CHECKING:
    import numpy as np
    from dask.array import Array
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers._utils.image.tiff import TiffMetadata
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.steinbock._layout import _SampleLayout

_IMAGE_AXES = ("c", "y", "x")
_LABEL_AXES = ("y", "x")
_MAX_EXACT_LABEL_BYTES = 4


@dataclass(frozen=True, slots=True)
class _SampleRasters:
    """Carry lazy rasters and the inspected frame used to validate cell metadata."""

    image: DataArray | DataTree
    labels: DataArray | DataTree
    label_shape: tuple[int, int]
    label_dtype: np.dtype[np.generic]


def _read_raster(path: ArtifactFile, *, target_axes: tuple[str, ...]) -> tuple[Array, TiffMetadata]:
    """Inspect one full-resolution series and retain semantic spatial axes lazily."""
    context = ReaderErrorContext("steinbock", path.name)
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
    try:
        array = normalize_raster_axes(
            raster.data,
            tuple(axis.lower() for axis in raster.axes),
            target_axes=target_axes,
            add_missing=("c",) if "c" in target_axes else (),
        )
    except ValueError as error:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=str(error),
            expected=f"scientific 2D axes {target_axes}, with only singleton extra dimensions",
        ) from error

    return array, metadata


def _read_rasters(sample: _SampleLayout, raster: RasterOptions) -> _SampleRasters:
    """Build aligned image and mask models in one sample's native pixel frame.

    One pyramid applies to each image and its mask, which share a pixel grid.
    """
    image, metadata = _read_raster(sample.image, target_axes=_IMAGE_AXES)
    labels, _ = _read_raster(sample.labels, target_axes=_LABEL_AXES)
    context = ReaderErrorContext("steinbock", sample.name)

    if image.shape[-2:] != labels.shape:
        raise ReaderFormatError(
            context=context,
            path=sample.labels,
            found=f"image YX {image.shape[-2:]} and mask {labels.shape}",
            expected="equal full-resolution image and mask extents",
        )
    if labels.dtype.kind not in {"i", "u"}:
        raise ReaderFormatError(
            context=context, path=sample.labels, found=f"dtype {labels.dtype}", expected="integer mask pixels"
        )
    factors = raster.scale_factors
    if factors and labels.dtype.itemsize > _MAX_EXACT_LABEL_BYTES:
        message = (
            "Generated label pyramids require <=32-bit integer labels. Upstream downsampling rounds 64-bit IDs. "
            "Use scale_factors=() for exact 64-bit masks."
        )
        raise ValueError(message)
    if factors and min(labels.shape) < prod(factors):
        message = f"Requested scale factors {factors} exceed the spatial extent of sample {sample.name!r}."
        raise ValueError(message)

    # OME channel identity is independent of both marker-table feature names and RGB.
    ome = metadata.ome
    if ome is None:
        raise ReaderFormatError(
            context=context, path=sample.image, found="no OME metadata", expected="a Steinbock OME-TIFF export"
        )
    channels = ome.images[0].pixels.channels
    if len(channels) != image.shape[0]:
        raise ReaderFormatError(
            context=context,
            path=sample.image,
            found=f"{len(channels)} OME channels and {image.shape[0]} raster channels",
            expected="one OME channel record per scientific channel",
        )

    names = [channel.name for channel in channels]
    channel_names = (
        [str(name) for name in names]
        if all(names) and len(set(names)) == len(names)
        else [channel.id for channel in channels]
    )
    if len(set(channel_names)) != len(channel_names):
        raise ReaderFormatError(
            context=context, path=sample.image, found="duplicate OME channel IDs", expected="unique channel identifiers"
        )

    return _SampleRasters(
        image=Image2DModel.parse(
            image,
            dims=_IMAGE_AXES,
            c_coords=channel_names,
            rgb=False,
            chunks=raster.image_chunks(),
            scale_factors=raster.model_scale_factors,
            transformations={sample.name: Identity()},
        ),
        labels=Labels2DModel.parse(
            labels,
            dims=_LABEL_AXES,
            chunks=raster.label_chunks(),
            scale_factors=raster.model_scale_factors,
            transformations={sample.name: Identity()},
        ),
        label_shape=(int(labels.shape[0]), int(labels.shape[1])),
        label_dtype=labels.dtype,
    )
