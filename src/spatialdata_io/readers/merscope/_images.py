"""Build one lazy ``(c, y, x)`` image per selected z plane from the mosaic TIFFs.

Each ``images/mosaic_<stain>_z<index>.tif`` holds one full-resolution
grayscale plane (singleton extra axes are allowed). The stains of one plane
are stacked in lexical order as channels named by the stain, keeping the
source dtype. All selected planes must share one spatial shape and dtype.

With ``micron_to_mosaic_pixel_transform.csv`` the images join the micrometre
``global`` frame through the inverse of that affine. Without it there is no
calibration, so the images keep their own ``mosaic_pixels`` frame (identity)
and no ``global`` transformation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import dask.array as da
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import inspect_tiff, normalize_raster_axes, read_tiff, select_tiff_series
from spatialdata_io.readers.merscope._transforms import read_transform

if TYPE_CHECKING:
    import numpy as np
    from dask.array import Array
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.merscope._layout import Mosaics

_IMAGE_AXES = ("c", "y", "x")
_CHANNEL_CONTEXT = ReaderErrorContext("merscope", "mosaic channel")
_GRID_CONTEXT = ReaderErrorContext("merscope", "mosaic image grid")


def _read_channel(path: ArtifactFile) -> Array:
    """Create one source graph from one inspected full-resolution grayscale plane."""
    metadata = inspect_tiff(path, context=_CHANNEL_CONTEXT)
    series = select_tiff_series(metadata, None, context=_CHANNEL_CONTEXT)
    raster = read_tiff(metadata, series=series, level=0, context=_CHANNEL_CONTEXT)
    try:
        array = normalize_raster_axes(raster.data, raster.axes, target_axes=("y", "x"))
    except ValueError as error:
        raise ReaderFormatError(
            context=_CHANNEL_CONTEXT,
            path=path,
            found=str(error),
            expected="a YX plane with only singleton nonspatial axes",
        ) from error
    if array.dtype.kind not in {"u", "i", "f"}:
        raise ReaderFormatError(
            context=_CHANNEL_CONTEXT,
            path=path,
            found=str(array.dtype),
            expected="real integer or floating-point grayscale pixels",
        )
    return array


def read_images(mosaics: Mosaics, *, dataset_id: str, options: RasterOptions) -> dict[str, DataArray | DataTree]:
    """Build one image ``<dataset_id>_z<index>`` per selected plane on a common source grid."""
    transformations: dict[str, BaseTransformation] = (
        {"mosaic_pixels": Identity()} if mosaics.transform is None else {"global": read_transform(mosaics.transform)}
    )
    images: dict[str, DataArray | DataTree] = {}
    reference: tuple[tuple[int, ...], np.dtype[np.generic]] | None = None
    for layer in mosaics.layers:
        arrays = [_read_channel(channel.path) for channel in layer.channels]
        for channel, array in zip(layer.channels, arrays, strict=True):
            if reference is None:
                # TIFF inspection provides finite integer dimensions; Dask's
                # general Array annotation also permits unknown NaN dimensions.
                reference = (cast("tuple[int, ...]", array.shape), array.dtype)
            elif (array.shape, array.dtype) != reference:
                raise ReaderFormatError(
                    context=_GRID_CONTEXT,
                    path=channel.path,
                    found=f"YX shape {array.shape} and dtype {array.dtype}",
                    expected=f"common shape {reference[0]} and dtype {reference[1]}",
                )
        image = da.stack(arrays, axis=0)
        options.validate_image_pyramid(image, element="a MERSCOPE mosaic")
        images[f"{dataset_id}_z{layer.z}"] = Image2DModel.parse(
            image,
            dims=_IMAGE_AXES,
            c_coords=[channel.stain for channel in layer.channels],
            rgb=False,
            scale_factors=options.model_scale_factors,
            chunks=options.image_chunks(),
            transformations=transformations,
        )
    return images
