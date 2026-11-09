"""Construct lazy MCMICRO images and labels in their exact local frames.

Every raster is one TIFF series read at full resolution (level zero):

- registered images (``registration/<image>.ome.tif``) and Coreograph cores
  (``dearray/<core>.ome.tif``) become ``c, y, x`` images in the source dtype.
  A missing ``C`` axis is added; any other non-spatial axis must be singleton.
  The channel count must equal the marker panel's length, and channels are
  named by the canonical marker IDs, never by OME channel names.
- segmentation masks (``segmentation/<method>-<image>/<mask>.ome.tif``) become
  integer ``y, x`` labels that must have the spatial shape of their source
  image. Label pixels are never read.
- Coreograph tissue masks (``dearray/masks/<core>_mask.tif``) are independent
  integer ``y, x`` labels on their own, possibly downsampled, grid.

Coordinate frames. Each raster keeps its native origin, the corner of pixel
``(0, 0)``. Coreograph records no authoritative crop origins, so no transform
between the registered map, its cores and the tissue masks is invented. A grid
has a calibration-independent name and is placed in exactly one frame: with
complete OME XY sizes a ``Scale`` into its micrometre frame, otherwise the
identity into ``<name>_pixels``:

- whole-slide image ``image``: ``global`` or ``image_pixels``;
- TMA map ``tma_map``: ``global`` or ``tma_map_pixels``;
- core ``core_<n>``: ``core_<n>`` or ``core_<n>_pixels``;
- tissue mask ``core_<n>_mask``: ``core_<n>_mask`` or ``core_<n>_mask_pixels``.

So ``global``, ``core_<n>`` and ``core_<n>_mask`` always mean micrometres and
``*_pixels`` always means source pixels. Calibration comes from the base OME
``PhysicalSizeX``/``PhysicalSizeY`` in µm, nm or mm; a partial declaration
leaves the grid in pixels. Segmentations share their image's frame and
transform: their own sizes may be absent, but a present size that disagrees
with the image's fails. A tissue mask uses only its own declaration. ASHLAR may
itself write a default spacing when acquisition metadata was missing, so a
declared size is source metadata, not a verified measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Identity, Scale

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    normalize_raster_axes,
    ome_pixel_sizes_um,
    read_tiff,
    require_matching_pixel_sizes,
    select_tiff_series,
)

if TYPE_CHECKING:
    import numpy as np
    from dask.array import Array
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers._utils.image.tiff import TiffMetadata
    from spatialdata_io.readers._utils.paths import ArtifactFile

_IMAGE_AXES = ("c", "y", "x")
_LABEL_AXES = ("y", "x")
_PIXEL_FRAME_SUFFIX = "_pixels"
_IMAGE_CONTEXT = ReaderErrorContext("mcmicro", "image")
_LABELS_CONTEXT = ReaderErrorContext("mcmicro", "labels")
_CALIBRATION_CONTEXT = ReaderErrorContext("mcmicro", "OME calibration")


@dataclass(frozen=True, slots=True)
class ImageGrid:
    """The inspected base extent, declared XY sizes and frame of one raster grid.

    ``declared_sizes_um`` keeps a partial declaration, so that same-grid
    segmentations can be checked against it; only a complete one calibrates.
    """

    name: str
    shape: tuple[int, int]
    declared_sizes_um: tuple[float | None, float | None]
    calibrated_frame: str

    @property
    def spacing_um(self) -> tuple[float, float] | None:
        """Return the complete XY spacing in micrometres per pixel, if declared."""
        x, y = self.declared_sizes_um
        return None if x is None or y is None else (x, y)

    @property
    def frame(self) -> str:
        """Return the micrometre frame when calibrated, else the grid's pixel frame."""
        return self.calibrated_frame if self.spacing_um is not None else f"{self.name}{_PIXEL_FRAME_SUFFIX}"

    @property
    def calibration(self) -> str:
        """Describe the calibration evidence recorded with physical positions."""
        return (
            "Source image OME PhysicalSizeX/PhysicalSizeY with declared source units; "
            f"XY={self.spacing_um!r} micrometer/pixel"
        )

    def transformations(self) -> dict[str, BaseTransformation]:
        """Map the grid to its single frame."""
        spacing = self.spacing_um
        return {self.frame: Identity() if spacing is None else Scale(list(spacing), axes=("x", "y"))}


def _read(path: ArtifactFile, context: ReaderErrorContext, target_axes: tuple[str, ...]) -> tuple[Array, TiffMetadata]:
    """Inspect once and normalize the full-resolution series to named 2D axes."""
    metadata = inspect_tiff(path, context=context)
    raster = read_tiff(metadata, series=select_tiff_series(metadata, None, context=context), level=0, context=context)
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


def _grid(name: str, calibrated_frame: str, array: Array, metadata: TiffMetadata) -> ImageGrid:
    x, y = ome_pixel_sizes_um(metadata, series=0, axes=("x", "y"), context=_CALIBRATION_CONTEXT)
    # TIFF inspection provides finite integer dimensions; Dask's general Array
    # annotation also permits unknown NaN dimensions.
    return ImageGrid(name, cast("tuple[int, int]", array.shape[-2:]), (x, y), calibrated_frame)


def _read_labels(path: ArtifactFile, options: RasterOptions) -> tuple[Array, TiffMetadata]:
    labels, metadata = _read(path, _LABELS_CONTEXT, _LABEL_AXES)
    if labels.dtype.kind not in {"i", "u"}:
        raise ReaderFormatError(
            context=_LABELS_CONTEXT, path=path, found=str(labels.dtype), expected="integer labels with background zero"
        )
    options.validate_label_pyramid(labels, element="an MCMICRO mask")
    return labels, metadata


def _labels_element(labels: Array, grid: ImageGrid, options: RasterOptions) -> DataArray | DataTree:
    return Labels2DModel.parse(
        labels,
        dims=_LABEL_AXES,
        scale_factors=options.model_scale_factors,
        chunks=options.label_chunks(),
        transformations=grid.transformations(),
    )


def read_image(
    path: ArtifactFile, *, name: str, calibrated_frame: str, channels: tuple[str, ...], options: RasterOptions
) -> tuple[DataArray | DataTree, ImageGrid]:
    """Return a lazy CYX image whose channels are the canonical marker panel, and its grid."""
    image, metadata = _read(path, _IMAGE_CONTEXT, _IMAGE_AXES)
    if image.shape[0] != len(channels):
        raise ReaderFormatError(
            context=ReaderErrorContext("mcmicro", "image channels"),
            path=path,
            found=f"{image.shape[0]} channels for {len(channels)} markers",
            expected="one registered channel per markers.csv row",
        )
    options.validate_image_pyramid(image, element="an MCMICRO raster")

    grid = _grid(name, calibrated_frame, image, metadata)
    element = Image2DModel.parse(
        image,
        dims=_IMAGE_AXES,
        c_coords=list(channels),
        rgb=False,
        scale_factors=options.model_scale_factors,
        chunks=options.image_chunks(),
        transformations=grid.transformations(),
    )
    return element, grid


def read_labels(
    path: ArtifactFile, *, grid: ImageGrid, options: RasterOptions
) -> tuple[DataArray | DataTree, np.dtype[np.integer]]:
    """Return lazy integer YX segmentation labels in the frame of their source image grid."""
    labels, metadata = _read_labels(path, options)
    if labels.shape != grid.shape:
        raise ReaderFormatError(
            context=_LABELS_CONTEXT,
            path=path,
            found=f"YX shape {labels.shape} for image grid {grid.shape}",
            expected="labels on the full-resolution grid of their source image",
        )

    # Present sizes must agree with the image's; absent ones are inherited.
    require_matching_pixel_sizes(
        ome_pixel_sizes_um(metadata, series=0, axes=("x", "y"), context=_CALIBRATION_CONTEXT),
        grid.declared_sizes_um,
        axes="xy",
        context=_CALIBRATION_CONTEXT,
        path=path,
        expected="segmentation calibration agreeing with its source image grid",
    )

    # _read_labels admitted only signed and unsigned integer dtypes.
    return _labels_element(labels, grid, options), cast("np.dtype[np.integer]", labels.dtype)


def read_mask(path: ArtifactFile, *, name: str, options: RasterOptions) -> DataArray | DataTree:
    """Return a lazy Coreograph tissue mask in its own frame, calibrated only by itself."""
    labels, metadata = _read_labels(path, options)
    return _labels_element(labels, _grid(name, name, labels, metadata), options)
