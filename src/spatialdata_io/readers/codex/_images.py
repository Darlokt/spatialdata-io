"""Lazy CODEX TIFF axes, channels, selection, and image models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import inspect_tiff, normalize_raster_axes, read_tiff

if TYPE_CHECKING:
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.paths import ArtifactFile

_IMAGE_AXES = ("c", "y", "x")
_MIN_PYRAMID_SIZE = 4


def _validate_image_options(
    data_axes: str | None, tiff_series: int | None, chunks: int | tuple[int, int, int] | None
) -> None:
    """Validate caller types, series selection, and positive CYX chunk cardinality."""
    if data_axes is not None and not isinstance(data_axes, str):
        message = "data_axes must be a string or None."
        raise TypeError(message)

    if tiff_series is not None and (
        isinstance(tiff_series, bool) or not isinstance(tiff_series, int) or tiff_series < 0
    ):
        message = "tiff_series must be a nonnegative integer or None."
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
    path: ArtifactFile, *, data_axes: str | None, tiff_series: int | None, chunks: int | tuple[int, int, int] | None
) -> DataArray | DataTree:
    """Build a level-zero image with declared axes and task-local TIFF ownership."""
    try:
        metadata = inspect_tiff(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=ReaderErrorContext("codex", "stitched image"), path=path)
        raise

    if tiff_series is None:
        if len(metadata.series) != 1:
            raise ReaderFormatError(
                context=ReaderErrorContext("codex", "stitched image"),
                path=path,
                found=f"{len(metadata.series)} series",
                expected="an explicit tiff_series selection",
            )
        tiff_series = 0

    raster = read_tiff(metadata, series=tiff_series, level=0)
    axes = tuple(axis.lower() for axis in (raster.axes if data_axes is None else data_axes))
    # TIFF samples represent RGB channels. Unknown stack axes require an override.
    axes = tuple("c" if axis == "s" else axis for axis in axes)
    image = normalize_raster_axes(raster.data, axes, target_axes=_IMAGE_AXES, add_missing=("c",))

    channel_names = None
    if metadata.ome is not None:
        channels = metadata.ome.images[tiff_series].pixels.channels
        if len(channels) == image.shape[0]:
            channel_names = [channel.name or channel.id for channel in channels]

    return Image2DModel.parse(
        image,
        dims=_IMAGE_AXES,
        c_coords=channel_names,
        scale_factors=[2, 2] if min(image.shape[-2:]) >= _MIN_PYRAMID_SIZE else None,
        chunks=tuple(max(axis_chunks) for axis_chunks in image.chunks) if chunks is None else chunks,
        rgb=False,
        transformations={"global": Identity()},
    )
