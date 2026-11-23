"""Read the ISS raw image and its labels lazily on one shared pixel grid.

Both artifacts are TIFF-family files with exactly one series; level zero (full
resolution) is read. Axes come from the file, or for the raw image from the
caller's ``data_axes``::

    raw image   source axes -> c, y, x   singleton nonspatial axes dropped, C added if absent
    labels      source axes -> y, x      integer dtype, only singleton extra axes

Channel ``C`` holds scientific channels and is never reinterpreted as RGB from
its count or dtype. An interleaved samples axis ``S`` becomes channels.
Declared axes that
cannot be normalized are malformed content (``ReaderFormatError``);
``data_axes`` that do not fit the file raise ``ValueError`` with a note naming
the raw image. Channel names, in order of precedence:

- ``r, g, b(, a)`` for samples the TIFF declares RGB (``PhotometricInterpretation``
  RGB or JPEG-compressed YCbCr; a fourth sample declared alpha by ``ExtraSamples``);
- OME channel names when all are nonempty and unique, else OME channel IDs,
  when the channel axis is the declared OME channel or samples axis; an OME
  channel expanded from samples names them ``<name or ID>:sample:<index>``;
- positional integers without OME metadata, or when ``data_axes`` reads another
  declared axis (for example Z) as ``C``.

The caller asserts that image and labels are registered on the same pixel grid;
only equal YX extents are checked. Complete OME ``PhysicalSizeX``/``Y`` of the
raw image (µm, nm or mm) give both elements a ``Scale`` to micrometres in
``global``. Labels inherit missing sizes, and a conflicting label declaration
fails. Without complete sizes, or when ``data_axes`` moves X or Y, both use the
identity to the shared pixel frame ``image_pixels`` instead, because ``global``
always means micrometres. Stage placement is never inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Identity, Scale

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    normalize_cyx_image,
    normalize_raster_axes,
    ome_pixel_sizes_um,
    read_tiff,
    require_matching_pixel_sizes,
    select_tiff_series,
)

if TYPE_CHECKING:
    import numpy as np
    from xarray import DataArray, DataTree

    from spatialdata_io.experimental.iss._layout import IssLayout
    from spatialdata_io.readers._utils.image import LazyRaster, RasterOptions
    from spatialdata_io.readers._utils.image.tiff import TiffMetadata
    from spatialdata_io.readers._utils.paths import ArtifactFile

_IMAGE_AXES = ("c", "y", "x")
_LABEL_AXES = ("y", "x")
_CALIBRATED_FRAME = "global"
_PIXEL_FRAME = "image_pixels"
_IMAGE_CONTEXT = ReaderErrorContext("iss", "raw image")
_LABELS_CONTEXT = ReaderErrorContext("iss", "labels")


@dataclass(frozen=True, slots=True)
class IssRasters:
    """Lazy image and labels models plus the label dtype that bounds instance IDs."""

    image: DataArray | DataTree
    labels: DataArray | DataTree
    label_dtype: np.dtype[np.integer]


def read_rasters(layout: IssLayout, *, data_axes: str | None, raster: RasterOptions) -> IssRasters:
    """Build the image and labels models without executing pixel tasks.

    One pyramid applies to both, because they share a pixel grid.
    """
    raw, image_metadata = _read_raster(layout.raw_image, context=_IMAGE_CONTEXT)
    image, channels = normalize_cyx_image(
        raw, data_axes, tiff=(image_metadata, 0), context=_IMAGE_CONTEXT, path=layout.raw_image
    )
    mask, label_metadata = _read_raster(layout.labels, context=_LABELS_CONTEXT)
    try:
        labels = normalize_raster_axes(mask.data, mask.axes, target_axes=_LABEL_AXES)
    except ValueError as error:
        raise ReaderFormatError(
            context=_LABELS_CONTEXT,
            path=layout.labels,
            found=str(error),
            expected="unambiguous 2D axes YX with only singleton nonspatial axes",
        ) from error

    if image.shape[-2:] != labels.shape:
        raise ReaderFormatError(
            context=ReaderErrorContext("iss", "raster alignment"),
            path=layout.labels,
            found=f"raw image {layout.raw_image} YX shape {image.shape[-2:]} and labels shape {labels.shape}",
            expected="the same full-resolution pixel extent for raw image and labels",
        )
    if labels.dtype.kind not in {"i", "u"}:
        raise ReaderFormatError(
            context=_LABELS_CONTEXT,
            path=layout.labels,
            found=f"dtype {labels.dtype}",
            expected="integer label pixels, with zero reserved for background",
        )
    raster.validate_label_pyramid(labels, element=f"ISS labels {layout.labels}")

    source_axes = tuple(axis.lower() for axis in raw.axes)
    scale = _micrometre_scale(image_metadata, label_metadata, source_axes, data_axes)
    frame, transform = (_PIXEL_FRAME, Identity()) if scale is None else (_CALIBRATED_FRAME, scale)
    return IssRasters(
        Image2DModel.parse(
            image,
            dims=_IMAGE_AXES,
            c_coords=channels,
            rgb=False,
            chunks=raster.image_chunks(),
            scale_factors=raster.model_scale_factors,
            transformations={frame: transform},
        ),
        Labels2DModel.parse(
            labels,
            dims=_LABEL_AXES,
            chunks=raster.label_chunks(),
            scale_factors=raster.model_scale_factors,
            transformations={frame: transform},
        ),
        labels.dtype,
    )


def _read_raster(path: ArtifactFile, *, context: ReaderErrorContext) -> tuple[LazyRaster, TiffMetadata]:
    """Inspect the only series and return its lazy full-resolution raster."""
    metadata = inspect_tiff(path, context=context)
    raster = read_tiff(metadata, series=select_tiff_series(metadata, None, context=context), level=0, context=context)
    return raster, metadata


def _micrometre_scale(
    image: TiffMetadata, labels: TiffMetadata, source_axes: tuple[str, ...], data_axes: str | None
) -> Scale | None:
    """Return the pair's scale to micrometres from the raw image's OME XY sizes, if known."""
    # OME sizes describe the declared X and Y; moving either axis leaves them unknown.
    if data_axes is not None and any(
        (declared in "xy" or given in "xy") and declared != given
        for declared, given in zip(source_axes, data_axes, strict=True)
    ):
        return None

    context = ReaderErrorContext("iss", "OME calibration")
    x, y = ome_pixel_sizes_um(image, series=0, axes=("x", "y"), context=context)
    # Labels inherit missing sizes; only two present declarations can conflict.
    require_matching_pixel_sizes(
        ome_pixel_sizes_um(labels, series=0, axes=("x", "y"), context=context),
        (x, y),
        axes="xy",
        context=context,
        path=labels.path,
        expected="label calibration agreeing with the caller-aligned raw image",
    )
    return Scale([x, y], axes=("x", "y")) if x is not None and y is not None else None
