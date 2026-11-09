"""Lazy CosMx FOV composite images and cell labels with their source pixels unchanged.

Rasters are read through the shared boundary by suffix: ``.jpg``/``.jpeg``
(whole-frame JPEG), ``.png`` (whole-frame PNG) and ``.tif``/``.tiff`` with exactly
one series, at full resolution::

    CellComposite/<prefix>_F<fov>.<jpg|jpeg|png|tif|tiff>   (c, y, x) image
    CellLabels/<prefix>_F<fov>.<tif|tiff>                  (y, x) integer labels

Interleaved samples (TIFF ``S``, PNG and JPEG components) become the image
channels, and a single-channel image gets a length-one ``c`` axis; any other
extra axis must be a singleton. Channels are ``r, g, b(, a)`` only when the
file declares RGB(A) samples (TIFF ``PhotometricInterpretation`` RGB or
JPEG-compressed YCbCr, PNG colour types 2, 3 and 6, three-component JPEG);
otherwise they are positional.

Pixels are never flipped: the reflection between raster rows and CosMx local
coordinates is part of each element's transformations (see ``_transforms``).
Labels must have the image's ``(y, x)`` grid when both exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Scale
from spatialdata.transformations import Sequence as TransformSequence

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    normalize_raster_axes,
    raster_channel_names,
    read_jpeg,
    read_png,
    read_tiff,
    select_tiff_series,
)
from spatialdata_io.readers.cosmx._schemas import IMAGES, LABELS
from spatialdata_io.readers.cosmx._transforms import fov_frame, global_frame, raster_to_local

if TYPE_CHECKING:
    from dask.array import Array
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import LazyRaster, RasterOptions
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.cosmx._transforms import Placement

_IMAGE_AXES = ("c", "y", "x")
_LABEL_AXES = ("y", "x")
_IMAGE_CONTEXT = ReaderErrorContext("cosmx", IMAGES)
_LABEL_CONTEXT = ReaderErrorContext("cosmx", LABELS)
_PLACEMENT_CONTEXT = ReaderErrorContext("cosmx", "FOV raster placement")


def _source(path: ArtifactFile, context: ReaderErrorContext) -> LazyRaster:
    """Read one full-resolution raster through the shared suffix-specific boundary."""
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return read_jpeg(path, context=context)
    if suffix == ".png":
        return read_png(path, context=context)
    metadata = inspect_tiff(path, context=context)
    return read_tiff(metadata, series=select_tiff_series(metadata, None, context=context), level=0, context=context)


def _normalize(
    data: Array, axes: tuple[str, ...], path: ArtifactFile, context: ReaderErrorContext, target: tuple[str, ...]
) -> Array:
    """Name source axes explicitly, dropping only singleton extra axes."""
    try:
        return normalize_raster_axes(data, axes, target_axes=target, add_missing=("c",) if "c" in target else ())
    except ValueError as error:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=str(error),
            expected=f"spatial axes {target} with only singleton additional axes",
        ) from error


def _frames(
    fov: int, height: int, placement: Placement, pixel_size_um: float | None, path: ArtifactFile
) -> dict[str, BaseTransformation]:
    """Map pixel centres to the FOV's local frame, then to the global frame."""
    local = raster_to_local(height)
    steps: list[BaseTransformation] = [local, placement.local_to_global()]
    if pixel_size_um is not None:
        steps.append(Scale([pixel_size_um, pixel_size_um], axes=("x", "y")))
    global_ = TransformSequence(steps)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        matrix = global_.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    if not np.isfinite(matrix).all():
        raise ReaderFormatError(
            context=_PLACEMENT_CONTEXT,
            path=path,
            found=f"FOV {fov} nonfinite global raster affine",
            expected="a finite affine after pixel-centre placement and calibration",
        )
    return {fov_frame(fov): local, global_frame(pixel_size_um): global_}


def read_image(
    path: ArtifactFile, *, fov: int, placement: Placement, options: RasterOptions, pixel_size_um: float | None = None
) -> tuple[DataArray | DataTree, tuple[int, int]]:
    """Return a lazy CYX FOV image and its trusted YX grid."""
    raster = _source(path, _IMAGE_CONTEXT)
    declared = tuple(axis.lower() for axis in raster.axes)
    # Interleaved samples are the image channels.
    axes = tuple("c" if axis == "s" else axis for axis in declared)
    image = _normalize(raster.data, axes, path, _IMAGE_CONTEXT, _IMAGE_AXES)
    options.validate_image_pyramid(image, element="a CosMx FOV raster")
    # Raster inspection provides finite integer dimensions; Dask also permits NaN dimensions.
    grid = cast("tuple[int, int]", image.shape[-2:])
    element = Image2DModel.parse(
        image,
        dims=_IMAGE_AXES,
        c_coords=raster_channel_names(raster.color, samples="s" in declared, ome=None),
        rgb=False,
        scale_factors=options.model_scale_factors,
        chunks=options.image_chunks(),
        transformations=_frames(fov, grid[0], placement, pixel_size_um, path),
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
    """Return lazy integer YX labels on the same grid as their FOV image, and their dtype."""
    raster = _source(path, _LABEL_CONTEXT)
    labels = _normalize(raster.data, raster.axes, path, _LABEL_CONTEXT, _LABEL_AXES)
    if grid is not None and labels.shape != grid:
        raise ReaderFormatError(
            context=_LABEL_CONTEXT, path=path, found=f"YX shape {labels.shape}", expected=f"the FOV image grid {grid}"
        )
    if labels.dtype.kind not in {"i", "u"}:
        raise ReaderFormatError(
            context=_LABEL_CONTEXT, path=path, found=str(labels.dtype), expected="integer labels with background zero"
        )
    options.validate_label_pyramid(labels, element="a CosMx FOV mask")

    element = Labels2DModel.parse(
        labels,
        dims=_LABEL_AXES,
        scale_factors=options.model_scale_factors,
        chunks=options.label_chunks(),
        transformations=_frames(fov, int(labels.shape[0]), placement, pixel_size_um, path),
    )
    return element, labels.dtype
