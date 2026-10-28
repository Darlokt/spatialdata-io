"""Build lazy registered images and mask labels through the shared TIFF boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import inspect_tiff, normalize_raster_axes, read_tiff

if TYPE_CHECKING:
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import LazyRaster, RasterOptions
    from spatialdata_io.readers.stereoseq._layout import ImageSource

_MAX_EXACT_LABEL_BYTES = 4


def _read(source: ImageSource) -> LazyRaster:
    """Read the single full-resolution series of one registered TIFF."""
    metadata = inspect_tiff(source.file)
    if len(metadata.series) != 1:
        raise ReaderFormatError(
            context=ReaderErrorContext("stereoseq", source.name),
            path=source.file,
            found=f"{len(metadata.series)} TIFF series",
            expected="one registered image series",
        )

    return read_tiff(metadata, series=0, level=0)


def read_image(source: ImageSource, *, raster: RasterOptions) -> DataArray | DataTree:
    """Read one 2D microscopy or expression image lazily as CYX."""
    lazy = _read(source)
    # TIFF samples carry RGB(A) semantics; scientific C axes retain channel order.
    axes = tuple("c" if axis.lower() == "s" else axis.lower() for axis in lazy.axes)
    data = normalize_raster_axes(lazy.data, axes, target_axes=("c", "y", "x"), add_missing=("c",))

    channel_count = int(data.shape[0])
    rgb_samples = "s" in tuple(axis.lower() for axis in lazy.axes) and channel_count in (3, 4)
    channels = ["r", "g", "b", "a"][:channel_count] if rgb_samples else None
    return Image2DModel.parse(
        data,
        dims=("c", "y", "x"),
        c_coords=channels,
        rgb=False,
        scale_factors=raster.model_scale_factors,
        chunks=raster.image_chunks(),
        transformations={"global": Identity()},
    )


def read_mask(source: ImageSource, *, raster: RasterOptions) -> DataArray | DataTree:
    """Read one single-channel mask lazily as YX labels with its source values.

    Labels pyramids downsample by nearest neighbour, so every level keeps only
    source mask values.
    """
    lazy = _read(source)
    # Masks have no channel semantics; a non-singleton C or S axis fails.
    data = normalize_raster_axes(lazy.data, lazy.axes, target_axes=("y", "x"))
    if data.dtype.kind not in "biu":
        raise ReaderFormatError(
            context=ReaderErrorContext("stereoseq", source.name),
            path=source.file,
            found=f"{data.dtype} pixels",
            expected="a boolean or integer mask",
        )
    if raster.scale_factors and data.dtype.itemsize > _MAX_EXACT_LABEL_BYTES:
        message = "Generated label pyramids require <=32-bit masks; use scale_factors=() for exact 64-bit values."
        raise ValueError(message)

    return Labels2DModel.parse(
        data,
        dims=("y", "x"),
        scale_factors=raster.model_scale_factors,
        chunks=raster.label_chunks(),
        transformations={"global": Identity()},
    )
