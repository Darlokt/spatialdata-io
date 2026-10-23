"""Construct lazy calibrated seqFISH DAPI and segmentation elements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isclose, prod
from typing import TYPE_CHECKING, cast

from spatialdata.models import Image2DModel, Labels2DModel

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import inspect_tiff, normalize_raster_axes, read_tiff
from spatialdata_io.readers.seqfish._transforms import pixel_scale

if TYPE_CHECKING:
    import numpy as np
    from dask.array import Array
    from ome_types.model import OME
    from spatialdata.transformations import Scale
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import TiffMetadata
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


def _inspect(path: ArtifactFile, artifact: str) -> TiffMetadata:
    """Add source context while preserving precise TIFF failure types."""
    try:
        return inspect_tiff(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=ReaderErrorContext("seqfish", artifact), path=path)
        raise


def _array(metadata: TiffMetadata, target_axes: tuple[str, ...]) -> Array:
    """Normalize full-resolution named axes without reading pixels."""
    context = ReaderErrorContext("seqfish", "raster axes")
    if len(metadata.series) != 1:
        raise ReaderFormatError(
            context=context,
            path=metadata.path,
            found=f"{len(metadata.series)} series",
            expected="exactly one full-resolution TIFF series",
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
            path=metadata.path,
            found=str(error),
            expected=f"2D spatial axes {target_axes} with only singleton additional axes",
        ) from error


@dataclass(frozen=True, slots=True)
class DapiRaster:
    """Own one normalized DAPI graph and its validated scientific calibration."""

    data: Array
    scale: Scale
    channels: tuple[str, ...]


def read_dapi(path: ArtifactFile) -> DapiRaster:
    """Inspect once and prepare one full-resolution CYX source graph."""
    metadata = _inspect(path, "DAPI")
    image = _array(metadata, _IMAGE_AXES)
    scale = pixel_scale(metadata)
    # pixel_scale establishes exactly one OME image before this private use.
    ome = cast("OME", metadata.ome)
    channels = ome.images[0].pixels.channels
    if len(channels) != image.shape[0]:
        raise ReaderFormatError(
            context=ReaderErrorContext("seqfish", "DAPI channels"),
            path=metadata.path,
            found=f"{len(channels)} OME channels for {image.shape[0]} raster channels",
            expected="one OME record per raster channel",
        )
    names = [channel.name for channel in channels]
    channel_names = (
        tuple(str(name) for name in names)
        if all(names) and len(set(names)) == len(names)
        else tuple(channel.id for channel in channels)
    )
    if len(set(channel_names)) != len(channel_names):
        raise ReaderFormatError(
            context=ReaderErrorContext("seqfish", "DAPI channels"),
            path=metadata.path,
            found=f"duplicate OME channel identifiers {channel_names!r}",
            expected="unique OME names or IDs",
        )
    return DapiRaster(image, scale, channel_names)


def _check_pyramid(array: Array, options: RasterOptions) -> None:
    """Reject pyramid factors that would eliminate a spatial dimension."""
    if options.scale_factors is not None and min(array.shape[-2:]) < prod(options.scale_factors):
        message = f"Requested scale factors {options.scale_factors} exceed the spatial extent of the seqFISH raster."
        raise ValueError(message)


def read_image(dapi: DapiRaster, *, roi: str, options: RasterOptions) -> DataArray | DataTree:
    """Model the prepared CYX graph without rebuilding its source tasks."""
    image = dapi.data
    _check_pyramid(image, options)
    return Image2DModel.parse(
        image,
        dims=_IMAGE_AXES,
        c_coords=list(dapi.channels),
        rgb=False,
        chunks=options.chunks or tuple(max(parts) for parts in image.chunks),
        scale_factors=options.scale_factors,
        transformations={roi: dapi.scale},
    )


def read_labels(
    path: ArtifactFile, *, reference: DapiRaster, roi: str, options: RasterOptions
) -> tuple[DataArray | DataTree, np.dtype[np.generic]]:
    """Return integer YX labels sharing the calibrated full-resolution DAPI grid."""
    metadata = _inspect(path, "segmentation")
    labels = _array(metadata, _LABEL_AXES)
    reference_shape = reference.data.shape[-2:]
    if labels.shape != reference_shape:
        raise ReaderFormatError(
            context=ReaderErrorContext("seqfish", "segmentation alignment"),
            path=path,
            found=f"segmentation YX shape {labels.shape} and DAPI shape {reference_shape}",
            expected="the same full-resolution pixel extent for segmentation and DAPI",
        )
    if metadata.ome is not None:
        pixels = metadata.ome.images[0].pixels
        if pixels.physical_size_x is not None or pixels.physical_size_y is not None:
            segmentation_scale = pixel_scale(metadata)
            if not all(
                isclose(actual, expected, rel_tol=1e-12)
                for actual, expected in zip(
                    segmentation_scale.to_scale_vector(("y", "x")),
                    reference.scale.to_scale_vector(("y", "x")),
                    strict=True,
                )
            ):
                raise ReaderFormatError(
                    context=ReaderErrorContext("seqfish", "segmentation calibration"),
                    path=path,
                    found="segmentation OME physical sizes differ from DAPI",
                    expected="the same calibrated pixel grid for segmentation and DAPI",
                )
    if labels.dtype.kind not in {"i", "u"}:
        raise ReaderFormatError(
            context=ReaderErrorContext("seqfish", "segmentation dtype"),
            path=path,
            found=str(labels.dtype),
            expected="integer label pixels with zero reserved for background",
        )
    if options.scale_factors is not None and labels.dtype.itemsize > _MAX_EXACT_LABEL_BYTES:
        message = "Generated label pyramids require <=32-bit labels; use scale_factors=None for exact 64-bit IDs."
        raise ValueError(message)
    _check_pyramid(labels, options)
    chunks = options.chunks
    label_chunks = chunks[-2:] if isinstance(chunks, tuple) else chunks
    return (
        Labels2DModel.parse(
            labels,
            dims=_LABEL_AXES,
            chunks=label_chunks or tuple(max(parts) for parts in labels.chunks),
            scale_factors=options.scale_factors,
            transformations={roi: reference.scale},
        ),
        labels.dtype,
    )
