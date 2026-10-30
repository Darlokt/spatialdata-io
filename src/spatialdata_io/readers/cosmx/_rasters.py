"""Lazy CosMx FOV images and labels with their source pixels left unchanged."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, cast

from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Scale
from spatialdata.transformations import Sequence as TransformSequence

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import inspect_tiff, normalize_raster_axes, read_jpeg, read_png, read_tiff
from spatialdata_io.readers.cosmx._transforms import raster_to_local

if TYPE_CHECKING:
    import numpy as np
    from dask.array import Array
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import LazyRaster, RasterOptions
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.cosmx._transforms import Placement

_IMAGE_AXES = ("c", "y", "x")
_LABEL_AXES = ("y", "x")
_COLOR_CHANNELS = {3: ["r", "g", "b"], 4: ["r", "g", "b", "a"]}
_MAX_EXACT_LABEL_BYTES = 4


def _source(path: ArtifactFile, artifact: str) -> LazyRaster:
    """Read one full-resolution raster through the shared suffix-specific boundary."""
    context = ReaderErrorContext("cosmx", artifact)
    suffix = path.suffix.lower()
    try:
        if suffix in {".jpg", ".jpeg"}:
            return read_jpeg(path)
        if suffix == ".png":
            return read_png(path)
        metadata = inspect_tiff(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=context, path=path)
        raise
    if len(metadata.series) != 1:
        raise ReaderFormatError(
            context=context, path=path, found=f"{len(metadata.series)} TIFF series", expected="one series"
        )
    return read_tiff(metadata, series=0, level=0)


def _normalize(raster: LazyRaster, path: ArtifactFile, artifact: str, target: tuple[str, ...]) -> Array:
    """Name source axes explicitly, dropping only singleton extra axes."""
    try:
        return normalize_raster_axes(
            raster.data, raster.axes, target_axes=target, add_missing=("c",) if target == _IMAGE_AXES else ()
        )
    except ValueError as error:
        raise ReaderFormatError(
            context=ReaderErrorContext("cosmx", artifact),
            path=path,
            found=str(error),
            expected=f"spatial axes {target} with only singleton additional axes",
        ) from error


def _frames(fov: int, height: int, placement: Placement, pixel_size_um: float | None) -> dict[str, BaseTransformation]:
    """Map pixel centers to the FOV's local frame, then to global pixels."""
    local = raster_to_local(height)
    steps: list[BaseTransformation] = [local, placement.local_to_global()]
    if pixel_size_um is not None:
        steps.append(Scale([pixel_size_um, pixel_size_um], axes=("x", "y")))
    return {str(fov): local, "global": TransformSequence(steps)}


def _check_pyramid(array: Array, options: RasterOptions) -> None:
    """Reject pyramid factors that would eliminate a spatial dimension."""
    if options.scale_factors and min(array.shape[-2:]) < prod(options.scale_factors):
        message = f"Requested scale factors {options.scale_factors} exceed the spatial extent of a CosMx FOV raster."
        raise ValueError(message)


def read_image(
    path: ArtifactFile, *, fov: int, placement: Placement, options: RasterOptions, pixel_size_um: float | None = None
) -> tuple[DataArray | DataTree, tuple[int, int]]:
    """Return a lazy CYX FOV image and its trusted YX grid."""
    image = _normalize(_source(path, "image"), path, "image", _IMAGE_AXES)
    _check_pyramid(image, options)
    # Raster inspection provides finite integer dimensions; Dask also permits NaN dimensions.
    grid = cast("tuple[int, int]", image.shape[-2:])
    element = Image2DModel.parse(
        image,
        dims=_IMAGE_AXES,
        c_coords=_COLOR_CHANNELS.get(int(image.shape[0])) if image.dtype.itemsize == 1 else None,
        rgb=False,
        scale_factors=options.model_scale_factors,
        chunks=options.image_chunks(),
        transformations=_frames(fov, grid[0], placement, pixel_size_um),
    )
    return element, grid


def read_labels(  # noqa: PLR0913, RUF100 - Independent raster/grid/placement state.
    path: ArtifactFile,
    *,
    fov: int,
    grid: tuple[int, int] | None,
    placement: Placement,
    options: RasterOptions,
    pixel_size_um: float | None = None,
) -> tuple[DataArray | DataTree, np.dtype[np.generic]]:
    """Return lazy integer YX labels on the same grid as their FOV image."""
    labels = _normalize(_source(path, "labels"), path, "labels", _LABEL_AXES)
    context = ReaderErrorContext("cosmx", "labels")
    if grid is not None and labels.shape != grid:
        raise ReaderFormatError(
            context=context, path=path, found=f"YX shape {labels.shape}", expected=f"the FOV image grid {grid}"
        )
    if labels.dtype.kind not in {"i", "u"}:
        raise ReaderFormatError(
            context=context, path=path, found=str(labels.dtype), expected="integer labels with background zero"
        )
    if options.scale_factors and labels.dtype.itemsize > _MAX_EXACT_LABEL_BYTES:
        message = "Generated label pyramids require <=32-bit labels; use scale_factors=() for exact 64-bit IDs."
        raise ValueError(message)
    _check_pyramid(labels, options)

    element = Labels2DModel.parse(
        labels,
        dims=_LABEL_AXES,
        scale_factors=options.model_scale_factors,
        chunks=options.label_chunks(),
        transformations=_frames(fov, int(labels.shape[0]), placement, pixel_size_um),
    )
    return element, labels.dtype
