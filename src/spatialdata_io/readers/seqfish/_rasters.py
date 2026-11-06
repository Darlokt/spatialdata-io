"""Construct lazy calibrated seqFISH DAPI and segmentation elements."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import TYPE_CHECKING, cast

import numpy as np
from spatialdata.models import Image2DModel, Labels2DModel

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import inspect_tiff, normalize_raster_axes, ome_channel_names, read_tiff
from spatialdata_io.readers.seqfish._transforms import pixel_scale

if TYPE_CHECKING:
    from dask.array import Array
    from spatialdata.transformations import Scale
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions, TiffMetadata
    from spatialdata_io.readers._utils.paths import ArtifactFile

_IMAGE_AXES = ("c", "y", "x")
_LABEL_AXES = ("y", "x")


def _inspect(path: ArtifactFile, artifact: str) -> TiffMetadata:
    """Add source context while preserving precise TIFF failure types."""
    metadata = inspect_tiff(path, context=ReaderErrorContext("seqfish", artifact))
    if len(metadata.series) != 1:
        raise ReaderFormatError(
            context=ReaderErrorContext("seqfish", artifact),
            path=metadata.path,
            found=f"{len(metadata.series)} series",
            expected="exactly one full-resolution TIFF series",
        )
    return metadata


def read_calibration(path: ArtifactFile) -> Scale:
    """Inspect DAPI physical XY calibration without constructing a pixel graph."""
    return pixel_scale(_inspect(path, "DAPI"))


def _array(metadata: TiffMetadata, target_axes: tuple[str, ...]) -> Array:
    """Normalize full-resolution named axes without reading pixels."""
    context = ReaderErrorContext("seqfish", "raster axes")
    raster = read_tiff(metadata, series=0, level=0, context=context)
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
    # Axis normalization already rejected a non-singleton samples axis.
    names = ome_channel_names(
        metadata,
        series=0,
        count=int(image.shape[0]),
        samples=False,
        context=ReaderErrorContext("seqfish", "DAPI channels"),
    )
    # pixel_scale establishes exactly one OME image before this private use.
    channel_names = tuple(cast("list[str]", names))
    return DapiRaster(image, scale, channel_names)


def read_image(dapi: DapiRaster, *, roi: str, options: RasterOptions) -> DataArray | DataTree:
    """Model the prepared CYX graph without rebuilding its source tasks."""
    image = dapi.data
    options.validate_image_pyramid(image, element="the seqFISH DAPI image")
    return Image2DModel.parse(
        image,
        dims=_IMAGE_AXES,
        c_coords=list(dapi.channels),
        rgb=False,
        scale_factors=options.model_scale_factors,
        chunks=options.image_chunks(),
        transformations={roi: dapi.scale},
    )


def read_labels(
    path: ArtifactFile, *, reference: DapiRaster, roi: str, options: RasterOptions, max_instance_id: int
) -> DataArray | DataTree:
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
    if max_instance_id > np.iinfo(labels.dtype.str).max:
        raise ReaderFormatError(
            context=ReaderErrorContext("seqfish", "label identity"),
            path=path,
            found="cell IDs exceeding the label dtype",
            expected=f"IDs representable by {labels.dtype}",
        )
    options.validate_label_pyramid(labels, element="the seqFISH segmentation")
    return Labels2DModel.parse(
        labels,
        dims=_LABEL_AXES,
        scale_factors=options.model_scale_factors,
        chunks=options.label_chunks(),
        transformations={roi: reference.scale},
    )
