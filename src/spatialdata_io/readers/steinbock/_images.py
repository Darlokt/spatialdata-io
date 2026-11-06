"""Read one sample's OME image and mask lazily and resolve their shared pixel frame and calibration.

Image ``ome/<sample>.ome.tiff`` and mask ``masks_<kind>/<sample>.tiff`` each
hold one TIFF series, read at full resolution (level zero):

- The image needs OME metadata. Its declared ``C``, ``Y`` and ``X`` axes are
  kept (a missing ``C`` is added); every other axis must be singleton and is
  dropped. Steinbock images are stacks of scientific marker channels, so a
  non-singleton samples axis ``S`` (an interleaved or planar RGB TIFF) is
  rejected rather than read as colour channels, and three or four channels are
  never treated as RGB. Channels keep their source order and are named by the
  OME channel names when all are nonempty and unique, else by the OME channel
  IDs; they are not matched against the table's feature names.
- The mask must be ``Y, X`` after dropping singleton axes, with an integer
  dtype and the image's ``Y, X`` extent. Its pixel values are the object IDs
  of the table; zero is background. Mask pixels are never read to validate IDs.

Coordinate frame: the image and its mask share one pixel grid whose origin is
the corner of pixel ``(0, 0)``, unchanged. A calibrated sample uses
``Scale(spacing_xy_um)`` into the micrometre frame ``calibrated_frame``
(``global`` when one sample is read, else the sample name). An uncalibrated
sample uses the identity into the pixel frame ``<sample>_pixels``, so no frame
name carries micrometres for one sample and pixels for another. Samples are
never registered to each other.

Calibration, in micrometres per pixel along X and Y:

1. ``images.csv`` acquisition extents divided by its ``width_px``/``height_px``
   when the image's row has both extents. The row's dimensions must equal the
   raster's ``Y, X`` extent; cropped or resized exports fail rather than being
   calibrated with stale dimensions.
2. Otherwise complete ``PhysicalSizeX``/``PhysicalSizeY`` of the image's OME
   metadata (µm, nm or mm).

Every present OME size of the image and mask, including a partial declaration,
must agree with the calibration in use (without one, the mask's with the
image's). An incomplete calibration leaves the sample in pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import TYPE_CHECKING

from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Identity, Scale

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    normalize_raster_axes,
    ome_channel_names,
    ome_pixel_sizes_um,
    read_tiff,
)
from spatialdata_io.readers.steinbock._layout import PIXEL_FRAME_SUFFIX

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from dask.array import Array
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions, TiffMetadata
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.steinbock._layout import SampleLayout
    from spatialdata_io.readers.steinbock._schemas import ImageMetadata

_IMAGE_AXES = ("c", "y", "x")
_LABEL_AXES = ("y", "x")
_IMAGE_CONTEXT = ReaderErrorContext("steinbock", "OME image")
_MASK_CONTEXT = ReaderErrorContext("steinbock", "mask")
_CALIBRATION_CONTEXT = ReaderErrorContext("steinbock", "sample calibration")
# Declarations of one spacing reach the reader through different arithmetic
# (an exact extent/width ratio here, a producer's float division or decimal
# text in OME), which can differ in the last bits. 1e-12 absorbs that and is
# far below any physically meaningful disagreement.
_SPACING_RTOL = 1e-12


@dataclass(frozen=True, slots=True)
class Calibration:
    """XY micrometres per pixel of one sample and the declaration that supplied them."""

    spacing_xy_um: tuple[float, float]
    source: str


@dataclass(frozen=True, slots=True)
class SampleGrid:
    """The pixel grid one sample's image, mask and table rows share.

    ``shape`` is the mask's ``(height, width)`` and ``dtype`` its integer dtype;
    ``frame`` is the coordinate system the rasters are registered to.
    """

    shape: tuple[int, int]
    dtype: np.dtype[np.integer]
    frame: str
    calibration: Calibration | None


@dataclass(frozen=True, slots=True)
class SampleRasters:
    """The lazy image and mask elements of one sample and their grid."""

    image: DataArray | DataTree
    labels: DataArray | DataTree
    grid: SampleGrid


def read_rasters(
    sample: SampleLayout, raster: RasterOptions, *, metadata: ImageMetadata | None, calibrated_frame: str
) -> SampleRasters:
    """Build the image and mask elements of one sample on their shared, possibly calibrated grid."""
    image, image_metadata = _read_raster(sample.image, target_axes=_IMAGE_AXES, context=_IMAGE_CONTEXT)
    labels, label_metadata = _read_raster(sample.labels, target_axes=_LABEL_AXES, context=_MASK_CONTEXT)

    if image.shape[-2:] != labels.shape:
        raise ReaderFormatError(
            context=_MASK_CONTEXT,
            path=sample.labels,
            found=f"image YX {image.shape[-2:]} and mask {labels.shape}",
            expected="equal full-resolution image and mask extents",
        )
    if labels.dtype.kind not in {"i", "u"}:
        raise ReaderFormatError(
            context=_MASK_CONTEXT, path=sample.labels, found=f"dtype {labels.dtype}", expected="integer mask pixels"
        )
    raster.validate_label_pyramid(labels, element=f"the mask of sample {sample.name!r}")

    # Axis normalization already rejected a non-singleton samples axis.
    channel_names = ome_channel_names(
        image_metadata, series=0, count=int(image.shape[0]), samples=False, context=_IMAGE_CONTEXT
    )
    if channel_names is None:
        raise ReaderFormatError(
            context=_IMAGE_CONTEXT, path=sample.image, found="no OME metadata", expected="a Steinbock OME-TIFF export"
        )

    shape = (int(labels.shape[0]), int(labels.shape[1]))
    calibration = _calibration(image_metadata, label_metadata, metadata, shape)
    if calibration is None:
        frame, transform = f"{sample.name}{PIXEL_FRAME_SUFFIX}", Identity()
    else:
        frame, transform = calibrated_frame, Scale(list(calibration.spacing_xy_um), axes=("x", "y"))
    return SampleRasters(
        image=Image2DModel.parse(
            image,
            dims=_IMAGE_AXES,
            c_coords=channel_names,
            rgb=False,
            chunks=raster.image_chunks(),
            scale_factors=raster.model_scale_factors,
            transformations={frame: transform},
        ),
        labels=Labels2DModel.parse(
            labels,
            dims=_LABEL_AXES,
            chunks=raster.label_chunks(),
            scale_factors=raster.model_scale_factors,
            transformations={frame: transform},
        ),
        grid=SampleGrid(shape, labels.dtype, frame, calibration),
    )


def _read_raster(
    path: ArtifactFile, *, target_axes: tuple[str, ...], context: ReaderErrorContext
) -> tuple[Array, TiffMetadata]:
    """Return the lazy full-resolution pixels of the only series in the target axes, and the metadata."""
    metadata = inspect_tiff(path, context=context)
    if len(metadata.series) != 1:
        raise ReaderFormatError(
            context=context, path=path, found=f"{len(metadata.series)} series", expected="exactly one TIFF series"
        )

    raster = read_tiff(metadata, series=0, level=0, context=context)
    try:
        array = normalize_raster_axes(
            raster.data,
            tuple(axis.lower() for axis in raster.axes),
            target_axes=target_axes,
            add_missing=("c",) if "c" in target_axes else (),
        )
    except ValueError as error:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=str(error),
            expected=f"scientific 2D axes {target_axes}, with only singleton extra dimensions",
        ) from error
    return array, metadata


def _calibration(
    image: TiffMetadata, labels: TiffMetadata, row: ImageMetadata | None, shape: tuple[int, int]
) -> Calibration | None:
    """Resolve the sample's XY spacing and check every other declaration against it."""
    if row is not None and row.shape != shape:
        raise ReaderFormatError(
            context=_CALIBRATION_CONTEXT,
            path=image.path,
            found=f"images.csv dimensions {row.shape} versus raster {shape}",
            expected="unchanged emitted image and mask dimensions matching images.csv",
        )
    image_sizes = ome_pixel_sizes_um(image, series=0, axes=("x", "y"), context=_CALIBRATION_CONTEXT)
    label_sizes = ome_pixel_sizes_um(labels, series=0, axes=("x", "y"), context=_CALIBRATION_CONTEXT)

    reference: tuple[float | None, ...]
    if row is not None and row.spacing_um is not None:
        _require_agreement(image_sizes, row.spacing_um, image.path)
        reference, source = (
            row.spacing_um,
            "images.csv acquisition_width_um/width_px and acquisition_height_um/height_px",
        )
    else:
        reference, source = image_sizes, "Source image OME PhysicalSizeX/PhysicalSizeY with declared source units"
    _require_agreement(label_sizes, reference, labels.path)

    x, y = reference
    if x is None or y is None:
        return None
    return Calibration((x, y), f"{source}; resolved XY={(x, y)!r} micrometer/pixel")


def _require_agreement(declared: tuple[float | None, ...], reference: tuple[float | None, ...], path: Path) -> None:
    """Reject two present XY sizes that disagree; an absent size never conflicts."""
    for axis, size, expected in zip("XY", declared, reference, strict=True):
        if size is not None and expected is not None and not isclose(size, expected, rel_tol=_SPACING_RTOL):
            raise ReaderFormatError(
                context=_CALIBRATION_CONTEXT,
                path=path,
                found=f"{axis} spacing {size} versus {expected} µm/pixel",
                expected="images.csv and linked image/mask calibration in agreement, including partial declarations",
            )
