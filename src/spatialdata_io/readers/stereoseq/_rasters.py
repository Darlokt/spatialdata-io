"""Read SAW registered images and masks lazily through the shared TIFF boundary.

SAW writes every image product as a single-series TIFF registered to the DNB
expression grid: one pixel is one bin-1 grid position, so images share the
coordinates of the count products and their transformations.

- ``<stain>_regist.tif`` (registered microscopy) and ``bin1_img.tif``
  (expression intensity) become ``(c, y, x)`` images with the source dtype and
  channel order. Their channels are ``r, g, b(, a)`` only when the TIFF
  declares an RGB colour model for its samples axis (the shared rule in
  ``readers._utils.image``); otherwise they are positional. A grayscale image
  gains one channel.
- ``*_mask.tif``, ``*_tissue_cut.tif``, ``*_mask_edm_dis_<d>.tif`` and
  ``bin1_img_tissue_cut.tif`` become ``(y, x)`` labels with the source boolean
  or integer values. A non-singleton channel or samples axis fails.

Declared axes that cannot become these axes (a non-singleton extra axis, or a
channel or samples axis in a mask) are malformed content and raise a
contextual ``ReaderFormatError``.

Only the first full-resolution level is read; pyramids are generated from
``scale_factors``, by nearest neighbour for labels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata.models import Image2DModel, Labels2DModel

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    normalize_raster_axes,
    raster_channel_names,
    read_tiff,
    select_tiff_series,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dask.array import Array
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import LazyRaster, RasterOptions
    from spatialdata_io.readers.stereoseq._layout import ImageSource


def _read(source: ImageSource) -> LazyRaster:
    """Read the single full-resolution series of one registered TIFF."""
    context = ReaderErrorContext("stereoseq", source.name)
    metadata = inspect_tiff(source.file, context=context)
    return read_tiff(metadata, series=select_tiff_series(metadata, None, context=context), level=0, context=context)


def _normalize(
    lazy: LazyRaster, source: ImageSource, axes: tuple[str, ...], *, target: tuple[str, ...], expected: str
) -> Array:
    """Normalize declared axes to ``target``; a failure is malformed content of the file."""
    try:
        return normalize_raster_axes(lazy.data, axes, target_axes=target, add_missing=("c",) if "c" in target else ())
    except ValueError as error:
        raise ReaderFormatError(
            context=ReaderErrorContext("stereoseq", source.name), path=source.file, found=str(error), expected=expected
        ) from error


def read_image(
    source: ImageSource, *, raster: RasterOptions, transformations: Mapping[str, BaseTransformation]
) -> DataArray | DataTree:
    """Read one 2D microscopy or expression image lazily as CYX."""
    lazy = _read(source)
    declared = tuple(axis.lower() for axis in lazy.axes)
    # Interleaved samples and scientific channels both become the channel axis.
    axes = tuple("c" if axis == "s" else axis for axis in declared)
    data = _normalize(
        lazy,
        source,
        axes,
        target=("c", "y", "x"),
        expected="Y and X axes with at most one channel or samples axis and only singleton other axes",
    )
    raster.validate_image_pyramid(data, element=f"Stereo-seq image {source.name!r}")

    return Image2DModel.parse(
        data,
        dims=("c", "y", "x"),
        c_coords=raster_channel_names(lazy.color, samples="s" in declared, ome=None),
        rgb=False,
        scale_factors=raster.model_scale_factors,
        chunks=raster.image_chunks(),
        transformations=dict(transformations),
    )


def read_mask(
    source: ImageSource, *, raster: RasterOptions, transformations: Mapping[str, BaseTransformation]
) -> DataArray | DataTree:
    """Read one single-channel mask lazily as YX labels with its source values."""
    lazy = _read(source)
    # Masks have no channel semantics; a non-singleton C or S axis fails.
    data = _normalize(
        lazy, source, lazy.axes, target=("y", "x"), expected="Y and X axes with only singleton other axes"
    )
    if data.dtype.kind not in "biu":
        raise ReaderFormatError(
            context=ReaderErrorContext("stereoseq", source.name),
            path=source.file,
            found=f"{data.dtype} pixels",
            expected="a boolean or integer mask",
        )
    raster.validate_label_pyramid(data, element=f"Stereo-seq mask {source.name!r}")

    return Labels2DModel.parse(
        data,
        dims=("y", "x"),
        scale_factors=raster.model_scale_factors,
        chunks=raster.label_chunks(),
        transformations=dict(transformations),
    )
