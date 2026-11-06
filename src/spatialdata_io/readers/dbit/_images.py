"""Normalize declared DBiT histology axes into lazy CYX images.

Channels are named ``r, g, b(, a)`` when the image's samples axis becomes the
channels and the file declares RGB samples (TIFF ``PhotometricInterpretation``
RGB or JPEG-compressed YCbCr, PNG colour types 2, 3 and 6, three-component
JPEG). Otherwise OME-TIFF channels keep their OME names or IDs (or
``<name or ID>:sample:<k>``), and other images positional indexes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    normalize_raster_axes,
    ome_channel_names,
    raster_channel_names,
    read_jpeg,
    read_png,
    read_tiff,
)

if TYPE_CHECKING:
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers._utils.paths import ArtifactFile

_IMAGE_AXES = ("c", "y", "x")
_TIFF_SUFFIXES = frozenset({".tif", ".tiff", ".btf"})


def _validate_image_options(data_axes: str | None, tiff_series: int, tiff_level: int) -> None:
    """Validate declared source axes and nonnegative TIFF selections."""
    if data_axes is not None:
        if not isinstance(data_axes, str):
            message = "data_axes must be a string or None."
            raise TypeError(message)

        axes = data_axes.lower().replace("s", "c")
        if not axes.isascii() or not axes.isalpha() or len(set(axes)) != len(axes) or not {"x", "y"}.issubset(axes):
            message = "data_axes must contain unique alphabetic axes including X and Y. S denotes samples as C."
            raise ValueError(message)

    for name, value in (("tiff_series", tiff_series), ("tiff_level", tiff_level)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            message = f"{name} must be a nonnegative integer."
            raise ValueError(message)


def _read_image(
    path: ArtifactFile,
    *,
    data_axes: str | None,
    tiff_series: int,
    tiff_level: int,
    raster: RasterOptions,
) -> DataArray | DataTree:
    """Read bounded metadata and construct an image without executing pixel tasks."""
    context = ReaderErrorContext("dbit", "histology image")
    if path.suffix.lower() not in _TIFF_SUFFIXES and (data_axes is not None or tiff_series != 0 or tiff_level != 0):
        message = "data_axes, tiff_series and tiff_level require a TIFF-family image."
        raise ValueError(message)

    # Shared loaders inspect metadata eagerly and retain only paths and
    # immutable selections in the lazy pixel graph.
    metadata = None
    if path.suffix.lower() in _TIFF_SUFFIXES:
        metadata = inspect_tiff(path, context=context)
        lazy = read_tiff(metadata, series=tiff_series, level=tiff_level, context=context)
    elif path.suffix.lower() == ".png":
        lazy = read_png(path, context=context)
    else:
        lazy = read_jpeg(path, context=context)

    # S represents interleaved samples (TIFF S, PNG and JPEG components), while
    # C denotes scientific channels.
    declared = tuple(axis.lower() for axis in lazy.axes)
    source = declared if data_axes is None else tuple(data_axes.lower())
    try:
        image = normalize_raster_axes(
            lazy.data,
            tuple("c" if axis == "s" else axis for axis in source),
            target_axes=_IMAGE_AXES,
            add_missing=("c",),
        )
    except ValueError as error:
        add_reader_context(error, context=context, path=path)
        raise

    # Declared RGB samples and OME sample records both apply only when the
    # file's samples axis became the channels.
    samples = "s" in declared and source[declared.index("s")] in {"c", "s"}
    ome = None
    if metadata is not None:
        ome = ome_channel_names(
            metadata, series=tiff_series, count=int(image.shape[0]), samples=samples, context=context
        )
    channel_names = raster_channel_names(lazy.color, samples=samples, ome=ome)

    return Image2DModel.parse(
        image,
        dims=_IMAGE_AXES,
        c_coords=channel_names,
        rgb=False,
        scale_factors=raster.model_scale_factors,
        chunks=raster.image_chunks(),
        transformations={"global": Identity()},
    )
