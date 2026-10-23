"""Normalize declared DBiT histology axes into lazy single-scale CYX images."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import inspect_tiff, normalize_raster_axes, read_jpeg, read_png, read_tiff

if TYPE_CHECKING:
    from xarray import DataArray

    from spatialdata_io.readers._utils.paths import ArtifactFile

_IMAGE_AXES = ("c", "y", "x")
_TIFF_SUFFIXES = frozenset({".tif", ".tiff", ".btf"})


def _validate_image_options(
    data_axes: str | None, tiff_series: int, tiff_level: int, chunks: int | tuple[int, int, int] | None
) -> None:
    """Validate axes, nonnegative TIFF selections, and positive CYX chunks."""
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

    if chunks is None:
        return

    sizes = chunks if isinstance(chunks, tuple) else (chunks,)
    if isinstance(chunks, tuple) and len(chunks) != len(_IMAGE_AXES):
        message = "chunks must contain three values in CYX order."
        raise ValueError(message)
    if any(isinstance(size, bool) or not isinstance(size, int) for size in sizes):
        message = "chunks must be an integer or a tuple of three integers."
        raise TypeError(message)
    if any(size <= 0 for size in sizes):
        message = "chunks must contain only positive sizes."
        raise ValueError(message)


def _read_image(
    path: ArtifactFile,
    *,
    data_axes: str | None,
    tiff_series: int,
    tiff_level: int,
    chunks: int | tuple[int, int, int] | None,
) -> DataArray:
    """Read bounded metadata and construct an image without executing pixel tasks."""
    context = ReaderErrorContext("dbit", "histology image")
    if path.suffix.lower() not in _TIFF_SUFFIXES and (data_axes is not None or tiff_series != 0 or tiff_level != 0):
        message = "data_axes, tiff_series and tiff_level require a TIFF-family image."
        raise ValueError(message)

    # Shared loaders inspect metadata eagerly and retain only paths and
    # immutable selections in the lazy pixel graph.
    metadata = None
    try:
        if path.suffix.lower() in _TIFF_SUFFIXES:
            metadata = inspect_tiff(path)
            raster = read_tiff(metadata, series=tiff_series, level=tiff_level)
        else:
            raster = read_png(path) if path.suffix.lower() == ".png" else read_jpeg(path)
    except (OSError, ValueError, IndexError) as error:
        add_reader_context(error, context=context, path=path)
        raise

    # TIFF S represents interleaved color samples, while C denotes scientific channels.
    axes = tuple(axis.lower() for axis in (raster.axes if data_axes is None else data_axes))
    samples = "s" in axes
    axes = tuple("c" if axis == "s" else axis for axis in axes)
    try:
        image = normalize_raster_axes(raster.data, axes, target_axes=_IMAGE_AXES, add_missing=("c",))
    except ValueError as error:
        add_reader_context(error, context=context, path=path)
        raise

    channel_names = None
    if metadata is not None and metadata.ome is not None and not samples:
        channels = metadata.ome.images[tiff_series].pixels.channels
        if len(channels) != image.shape[0]:
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"{len(channels)} OME channel records for {image.shape[0]} output channels",
                expected="one OME channel record per scientific channel",
            )

        # OME IDs provide stable labels when display names are missing or repeated.
        names = [channel.name for channel in channels]
        channel_names = (
            [str(name) for name in names]
            if all(names) and len(set(names)) == len(names)
            else [channel.id for channel in channels]
        )

    # The parser annotates a union even though scale_factors=None always returns a DataArray.
    return cast(
        "DataArray",
        Image2DModel.parse(
            image,
            dims=_IMAGE_AXES,
            c_coords=channel_names,
            rgb=False,
            chunks=tuple(max(parts) for parts in image.chunks) if chunks is None else chunks,
            transformations={"global": Identity()},
        ),
    )
