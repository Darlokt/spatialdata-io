"""Construct lazy MCMICRO images and labels in their exact local pixel frames."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, prod
from typing import TYPE_CHECKING, cast

from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Identity, Scale

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import inspect_tiff, normalize_raster_axes, ome_pixel_sizes_um, read_tiff

if TYPE_CHECKING:
    import numpy as np
    from dask.array import Array
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers._utils.image.tiff import TiffMetadata
    from spatialdata_io.readers._utils.paths import ArtifactFile

_IMAGE_AXES = ("c", "y", "x")
_LABEL_AXES = ("y", "x")
_MAX_EXACT_LABEL_BYTES = 4
_CALIBRATION = ReaderErrorContext("mcmicro", "OME calibration")


@dataclass(frozen=True, slots=True)
class ImageGrid:
    """Own the inspected base image extent and optional local XY spacing."""

    shape: tuple[int, int]
    declared_sizes_um: tuple[float | None, float | None]

    def transform(self) -> Identity | Scale:
        """Scale to micrometres only when both source axes are calibrated."""
        x, y = self.declared_sizes_um
        return Scale([x, y], axes=("x", "y")) if x is not None and y is not None else Identity()


def _read(path: ArtifactFile, artifact: str, target_axes: tuple[str, ...]) -> tuple[Array, TiffMetadata]:
    """Inspect once and normalize the full-resolution series to named 2D axes."""
    context = ReaderErrorContext("mcmicro", artifact)
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
            expected="one full-resolution series",
        )

    raster = read_tiff(metadata, series=0, level=0)
    try:
        array = normalize_raster_axes(
            raster.data,
            raster.axes,
            target_axes=target_axes,
            add_missing=("c",) if target_axes == _IMAGE_AXES else (),
        )
    except ValueError as error:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=str(error),
            expected=f"spatial axes {target_axes} with only singleton additional axes",
        ) from error

    return array, metadata


def _check_pyramid(array: Array, options: RasterOptions) -> None:
    """Reject pyramid factors that would eliminate a spatial dimension."""
    if options.scale_factors and min(array.shape[-2:]) < prod(options.scale_factors):
        message = f"Requested scale factors {options.scale_factors} exceed the spatial extent of an MCMICRO raster."
        raise ValueError(message)


def _declared_sizes(metadata: TiffMetadata) -> tuple[float | None, float | None]:
    x, y = ome_pixel_sizes_um(metadata, series=0, axes=("x", "y"), context=_CALIBRATION)
    return x, y


def read_image(
    path: ArtifactFile, *, channels: tuple[str, ...], frame: str, options: RasterOptions
) -> tuple[DataArray | DataTree, ImageGrid]:
    """Return a lazy CYX image whose channels are the canonical marker panel."""
    image, metadata = _read(path, "image", _IMAGE_AXES)
    if image.shape[0] != len(channels):
        raise ReaderFormatError(
            context=ReaderErrorContext("mcmicro", "image channels"),
            path=path,
            found=f"{image.shape[0]} channels for {len(channels)} markers",
            expected="one registered channel per markers.csv row",
        )
    _check_pyramid(image, options)

    # TIFF inspection provides finite integer dimensions; Dask's general Array
    # annotation also permits unknown NaN dimensions.
    grid = ImageGrid(cast("tuple[int, int]", image.shape[-2:]), _declared_sizes(metadata))
    element = Image2DModel.parse(
        image,
        dims=_IMAGE_AXES,
        c_coords=list(channels),
        rgb=False,
        scale_factors=options.model_scale_factors,
        chunks=options.image_chunks(),
        transformations={frame: grid.transform()},
    )
    return element, grid


def read_labels(
    path: ArtifactFile, *, grid: ImageGrid | None, frame: str, options: RasterOptions
) -> tuple[DataArray | DataTree, np.dtype[np.generic]]:
    """Return lazy integer YX labels, requiring the image grid when one is given."""
    labels, metadata = _read(path, "labels", _LABEL_AXES)
    context = ReaderErrorContext("mcmicro", "labels")
    if grid is not None and labels.shape != grid.shape:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"YX shape {labels.shape} for image grid {grid.shape}",
            expected="labels on the full-resolution grid of their source image",
        )
    if labels.dtype.kind not in {"i", "u"}:
        raise ReaderFormatError(
            context=context, path=path, found=str(labels.dtype), expected="integer labels with background zero"
        )
    if options.scale_factors and labels.dtype.itemsize > _MAX_EXACT_LABEL_BYTES:
        message = "Generated label pyramids require <=32-bit labels; use scale_factors=() for exact 64-bit IDs."
        raise ValueError(message)
    _check_pyramid(labels, options)

    declared = _declared_sizes(metadata)
    if grid is None:
        # An independent tissue mask may only use its own complete declaration.
        grid = ImageGrid(cast("tuple[int, int]", labels.shape), declared)
    else:
        # Same-grid segmentations inherit missing sizes; present ones must agree.
        for axis, expected, size in zip("XY", grid.declared_sizes_um, declared, strict=True):
            if expected is not None and size is not None and not isclose(size, expected, rel_tol=1e-12):
                raise ReaderFormatError(
                    context=_CALIBRATION,
                    path=metadata.path,
                    found=f"{axis} spacing {size} versus {expected} µm",
                    expected="segmentation calibration agreeing with its source image grid",
                )
    element = Labels2DModel.parse(
        labels,
        dims=_LABEL_AXES,
        scale_factors=options.model_scale_factors,
        chunks=options.label_chunks(),
        transformations={frame: grid.transform()},
    )
    return element, labels.dtype
