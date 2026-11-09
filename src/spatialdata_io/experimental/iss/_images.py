"""Read the ISS raw image and its labels lazily on one shared pixel grid.

Both artifacts are TIFF-family files with exactly one series; level zero (full
resolution) is read. Axes come from the file, or for the raw image from the
caller's ``data_axes``::

    raw image   source axes -> c, y, x   singleton nonspatial axes dropped, C added if absent
    labels      source axes -> y, x      integer dtype, only singleton extra axes

Channel ``C`` holds scientific channels and is never reinterpreted as RGB from
its count or dtype. A non-singleton samples axis ``S`` (interleaved RGB, for
example) is rejected unless ``data_axes`` maps it to ``C``. Declared axes that
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

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import (
    channel_source,
    inspect_tiff,
    normalize_raster_axes,
    ome_channel_names,
    ome_pixel_sizes_um,
    raster_channel_names,
    read_tiff,
    require_matching_pixel_sizes,
    select_tiff_series,
)

if TYPE_CHECKING:
    import numpy as np
    from dask.array import Array
    from xarray import DataArray, DataTree

    from spatialdata_io.experimental.iss._layout import IssLayout
    from spatialdata_io.readers._utils.image import RasterOptions
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


def validate_data_axes(data_axes: str | None) -> str | None:
    """Validate the caller's raw-image axes before any I/O and return them lowercase.

    The rank is checked against the file when it is read.
    """
    if data_axes is None:
        return None
    if not isinstance(data_axes, str):
        message = "data_axes must be a string such as 'CYX', or None."
        raise TypeError(message)

    axes = data_axes.lower()
    if not axes.isascii() or not axes.isalpha() or len(set(axes)) != len(axes) or not {"x", "y"} <= set(axes):
        message = f"data_axes must contain distinct letters including X and Y, such as 'CYX'; found {data_axes!r}."
        raise ValueError(message)
    return axes


def read_rasters(layout: IssLayout, *, data_axes: str | None, raster: RasterOptions) -> IssRasters:
    """Build the image and labels models without executing pixel tasks.

    One pyramid applies to both, because they share a pixel grid.
    """
    image, image_metadata = _read_raster(
        layout.raw_image, context=_IMAGE_CONTEXT, target_axes=_IMAGE_AXES, data_axes=data_axes
    )
    labels, label_metadata = _read_raster(layout.labels, context=_LABELS_CONTEXT, target_axes=_LABEL_AXES)

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

    # A declared S becomes channels only when data_axes maps it to C, because
    # normalization rejects a non-singleton S otherwise.
    level = image_metadata.series[0].levels[0]
    source_axes = tuple(axis.lower() for axis in level.axes)
    samples, ome_samples = channel_source(source_axes, source_axes if data_axes is None else data_axes)
    ome = None
    if ome_samples is not None:
        ome = ome_channel_names(
            image_metadata,
            series=0,
            count=int(image.shape[0]),
            samples=ome_samples,
            context=ReaderErrorContext("iss", "raw image channels"),
        )

    scale = _micrometre_scale(image_metadata, label_metadata, source_axes, data_axes)
    frame, transform = (_PIXEL_FRAME, Identity()) if scale is None else (_CALIBRATED_FRAME, scale)
    return IssRasters(
        Image2DModel.parse(
            image,
            dims=_IMAGE_AXES,
            c_coords=raster_channel_names(level.color, samples=samples, ome=ome),
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


def _read_raster(
    path: ArtifactFile, *, context: ReaderErrorContext, target_axes: tuple[str, ...], data_axes: str | None = None
) -> tuple[Array, TiffMetadata]:
    """Inspect the only series and normalize its full-resolution level lazily to ``target_axes``."""
    metadata = inspect_tiff(path, context=context)
    raster = read_tiff(metadata, series=select_tiff_series(metadata, None, context=context), level=0, context=context)
    axes = tuple(data_axes) if data_axes is not None else tuple(axis.lower() for axis in raster.axes)
    try:
        array = normalize_raster_axes(
            raster.data, axes, target_axes=target_axes, add_missing=("c",) if "c" in target_axes else ()
        )
    except ValueError as error:
        if data_axes is not None:
            add_reader_context(error, context=context, path=path)
            raise
        raise ReaderFormatError(
            context=context,
            path=path,
            found=str(error),
            expected=f"unambiguous 2D axes {target_axes}. Singleton nonspatial axes only",
        ) from error
    return array, metadata


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
