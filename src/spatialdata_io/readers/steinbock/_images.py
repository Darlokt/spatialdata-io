"""Interpret Steinbock OME channels and construct lazy per-sample rasters."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, prod
from typing import TYPE_CHECKING

from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Identity, Scale

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import inspect_tiff, normalize_raster_axes, ome_pixel_sizes_um, read_tiff

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from dask.array import Array
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers._utils.image.tiff import TiffMetadata
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.steinbock._layout import _SampleLayout
    from spatialdata_io.readers.steinbock._schemas import ImageMetadata

_IMAGE_AXES = ("c", "y", "x")
_CALIBRATION = ReaderErrorContext("steinbock", "sample calibration")
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


def _require_agreement(declared: tuple[float | None, ...], reference: tuple[float | None, ...], path: Path) -> None:
    """Reject two present XY sizes that disagree; an absent size never conflicts."""
    for axis, size, expected in zip("XY", declared, reference, strict=True):
        if size is not None and expected is not None and not isclose(size, expected, rel_tol=1e-12):
            raise ReaderFormatError(
                context=_CALIBRATION,
                path=path,
                found=f"{axis} spacing {size} versus {expected} µm/pixel",
                expected="images.csv and linked image/mask calibration in agreement, including partial declarations",
            )


def _spacing(
    image: TiffMetadata, labels: TiffMetadata, metadata: ImageMetadata | None, shape: tuple[int, int]
) -> tuple[float, float] | None:
    """Resolve IMC extents and per-artifact OME declarations on an unchanged sample grid."""
    if metadata is not None and metadata.shape != shape:
        raise ReaderFormatError(
            context=_CALIBRATION,
            path=image.path,
            found=f"images.csv dimensions {metadata.shape} versus raster {shape}",
            expected="unchanged emitted image and mask dimensions matching images.csv",
        )
    image_sizes = ome_pixel_sizes_um(image, series=0, axes=("x", "y"), context=_CALIBRATION)
    label_sizes = ome_pixel_sizes_um(labels, series=0, axes=("x", "y"), context=_CALIBRATION)

    # IMC acquisition extents take precedence; complete image OME sizes are the alternative.
    x, y = image_sizes
    spacing = metadata.spacing_um if metadata is not None else None
    if spacing is None and x is not None and y is not None:
        spacing = (x, y)

    # Every present declaration must agree with the resolved calibration, or
    # without one, mask declarations with the image's partial declarations.
    if spacing is not None:
        _require_agreement(image_sizes, spacing, image.path)
    _require_agreement(label_sizes, spacing if spacing is not None else image_sizes, labels.path)
    return spacing


def _read_rasters(
    sample: _SampleLayout, raster: RasterOptions, *, metadata: ImageMetadata | None, calibrated_frame: str
) -> _SampleRasters:
    """Build aligned image and mask models for one sample.

    One pyramid applies to each image and its mask, which share a pixel grid.
    Calibrated pairs use ``calibrated_frame`` in micrometres; uncalibrated pairs
    keep their native pixel frame named after the sample.
    """
    image, image_metadata = _read_raster(sample.image, target_axes=_IMAGE_AXES)
    labels, label_metadata = _read_raster(sample.labels, target_axes=_LABEL_AXES)
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
    ome = image_metadata.ome
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

    spacing = _spacing(image_metadata, label_metadata, metadata, (int(labels.shape[0]), int(labels.shape[1])))
    transform = Scale(list(spacing), axes=("x", "y")) if spacing is not None else Identity()
    coordinate_system = calibrated_frame if spacing is not None else sample.name
    return _SampleRasters(
        image=Image2DModel.parse(
            image,
            dims=_IMAGE_AXES,
            c_coords=channel_names,
            rgb=False,
            chunks=raster.image_chunks(),
            scale_factors=raster.model_scale_factors,
            transformations={coordinate_system: transform},
        ),
        labels=Labels2DModel.parse(
            labels,
            dims=_LABEL_AXES,
            chunks=raster.label_chunks(),
            scale_factors=raster.model_scale_factors,
            transformations={coordinate_system: transform},
        ),
        label_shape=(int(labels.shape[0]), int(labels.shape[1])),
        label_dtype=labels.dtype,
    )
