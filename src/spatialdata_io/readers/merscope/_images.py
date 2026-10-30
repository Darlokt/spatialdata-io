"""Build lazy full-resolution CYX mosaics with deterministic scientific channels."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, cast

import dask.array as da
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import inspect_tiff, normalize_raster_axes, read_tiff

if TYPE_CHECKING:
    import numpy as np
    from dask.array import Array
    from spatialdata.transformations import Affine
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.merscope._layout import MosaicLayer

_IMAGE_AXES = ("c", "y", "x")


def _read_channel(path: ArtifactFile) -> Array:
    """Create one source graph from one inspected full-resolution grayscale plane."""
    context = ReaderErrorContext("merscope", "mosaic channel")
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
            expected="one full-resolution grayscale series",
        )
    raster = read_tiff(metadata, series=0, level=0)
    try:
        array = normalize_raster_axes(raster.data, raster.axes, target_axes=("y", "x"))
    except ValueError as error:
        raise ReaderFormatError(
            context=context, path=path, found=str(error), expected="a YX plane with only singleton nonspatial axes"
        ) from error
    if array.dtype.kind not in {"u", "i", "f"}:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=str(array.dtype),
            expected="real integer or floating-point grayscale pixels",
        )
    return array


def read_images(
    layers: tuple[MosaicLayer, ...],
    *,
    dataset_id: str,
    options: RasterOptions,
    transform: Affine | None,
    frame: str,
) -> dict[str, DataArray | DataTree]:
    """Build one image per requested z index and validate a common source grid.

    ``transform`` maps mosaic pixels into micrometres; without it, ``frame``
    is the mosaic pixel grid itself.
    """
    images: dict[str, DataArray | DataTree] = {}
    reference: tuple[tuple[int, ...], np.dtype[np.generic]] | None = None
    for layer in layers:
        arrays = [_read_channel(channel.path) for channel in layer.channels]
        for channel, array in zip(layer.channels, arrays, strict=True):
            if reference is None:
                # TIFF inspection provides finite integer dimensions; Dask's
                # general Array annotation also permits unknown NaN dimensions.
                reference = (cast("tuple[int, ...]", array.shape), array.dtype)
            elif (array.shape, array.dtype) != reference:
                raise ReaderFormatError(
                    context=ReaderErrorContext("merscope", "mosaic grid"),
                    path=channel.path,
                    found=f"YX shape {array.shape} and dtype {array.dtype}",
                    expected=f"common shape {reference[0]} and dtype {reference[1]}",
                )
        image = da.stack(arrays, axis=0)
        if options.scale_factors and min(image.shape[-2:]) < prod(options.scale_factors):
            message = "Requested scale_factors exceed the spatial extent of a MERSCOPE mosaic."
            raise ValueError(message)
        images[f"{dataset_id}_z{layer.z}"] = Image2DModel.parse(
            image,
            dims=_IMAGE_AXES,
            c_coords=[channel.stain for channel in layer.channels],
            rgb=False,
            scale_factors=options.model_scale_factors,
            chunks=options.image_chunks(),
            transformations={frame: Identity() if transform is None else transform},
        )
    return images
