"""Construct the lazy seqFISH DAPI image and segmentation labels of one ROI.

Both rasters are single-series TIFFs read at full resolution:

- ``<ROI>_DAPI.tiff``: an OME-TIFF image with physical pixel sizes (see
  :mod:`._transforms`) and spatial axes ``YX`` plus optional singleton axes
  or a channel axis ``C``. It becomes a ``(c, y, x)`` image whose channel
  coordinates are the OME channel names, or their IDs when names are missing
  or repeated.
- ``<ROI>_Segmentation.tiff``: integer ``(y, x)`` cell labels, ``0`` for
  background and otherwise the cell ID, on the DAPI pixel grid. It must have
  the DAPI shape, and every OME physical size it declares must equal the DAPI
  one; a size it does not declare is inherited from DAPI.

Both are mapped into the ROI's micrometre frame with the DAPI pixel scale.
Pixels are not read: the label dtype is checked against the largest table cell
ID instead of scanning label values. A non-singleton samples axis (RGB) is
rejected, because DAPI and labels are single-component rasters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from spatialdata.models import Image2DModel, Labels2DModel

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    normalize_raster_axes,
    ome_channel_names,
    ome_pixel_sizes_um,
    read_tiff,
    require_matching_pixel_sizes,
    select_tiff_series,
)
from spatialdata_io.readers.seqfish._transforms import pixel_scale

if TYPE_CHECKING:
    from dask.array import Array
    from spatialdata.transformations import Scale
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions, TiffMetadata
    from spatialdata_io.readers._utils.paths import ArtifactFile

_DAPI_CONTEXT = ReaderErrorContext("seqfish", "DAPI")
_SEGMENTATION_CONTEXT = ReaderErrorContext("seqfish", "Segmentation")
_IMAGE_AXES = ("c", "y", "x")
_LABEL_AXES = ("y", "x")


def _inspect(path: ArtifactFile, context: ReaderErrorContext) -> TiffMetadata:
    """Inspect a TIFF once and require exactly one series, read at full resolution."""
    metadata = inspect_tiff(path, context=context)
    select_tiff_series(metadata, None, context=context)
    return metadata


def read_calibration(path: ArtifactFile) -> Scale:
    """Inspect DAPI physical XY calibration without constructing a pixel graph."""
    return pixel_scale(_inspect(path, _DAPI_CONTEXT), context=_DAPI_CONTEXT)


def _array(metadata: TiffMetadata, target_axes: tuple[str, ...], context: ReaderErrorContext) -> Array:
    """Normalize full-resolution named axes without reading pixels."""
    raster = read_tiff(metadata, series=0, level=0, context=context)
    try:
        return normalize_raster_axes(
            raster.data,
            raster.axes,
            target_axes=target_axes,
            add_missing=("c",) if target_axes == _IMAGE_AXES else (),
        )
    except ValueError as error:
        raise ReaderFormatError(
            context=context,
            path=metadata.path,
            found=str(error),
            expected=f"2D spatial axes {target_axes} with only singleton additional axes",
        ) from error


@dataclass(frozen=True, slots=True)
class DapiRaster:
    """One normalized DAPI graph with its validated calibration and channel names."""

    data: Array
    scale: Scale
    channels: tuple[str, ...]


def read_dapi(path: ArtifactFile) -> DapiRaster:
    """Inspect once and prepare one full-resolution CYX source graph."""
    metadata = _inspect(path, _DAPI_CONTEXT)
    image = _array(metadata, _IMAGE_AXES, _DAPI_CONTEXT)
    scale = pixel_scale(metadata, context=_DAPI_CONTEXT)
    # Axis normalization already rejected a non-singleton samples axis.
    names = ome_channel_names(metadata, series=0, count=int(image.shape[0]), samples=False, context=_DAPI_CONTEXT)
    # pixel_scale establishes exactly one OME image, so names are not None.
    return DapiRaster(image, scale, tuple(cast("list[str]", names)))


def read_image(dapi: DapiRaster, *, frame: str, options: RasterOptions) -> DataArray | DataTree:
    """Model the prepared CYX graph without rebuilding its source tasks."""
    options.validate_image_pyramid(dapi.data, element="the seqFISH DAPI image")
    return Image2DModel.parse(
        dapi.data,
        dims=_IMAGE_AXES,
        c_coords=list(dapi.channels),
        rgb=False,
        scale_factors=options.model_scale_factors,
        chunks=options.image_chunks(),
        transformations={frame: dapi.scale},
    )


def _check_calibration(metadata: TiffMetadata, reference: Scale) -> None:
    """Require every declared segmentation pixel size to equal the DAPI one; absent sizes are inherited."""
    require_matching_pixel_sizes(
        ome_pixel_sizes_um(metadata, series=0, axes=("y", "x"), context=_SEGMENTATION_CONTEXT),
        [float(size) for size in reference.to_scale_vector(("y", "x"))],
        axes="yx",
        context=_SEGMENTATION_CONTEXT,
        path=metadata.path,
        expected="the same calibrated pixel grid for segmentation and DAPI",
    )


def read_labels(
    path: ArtifactFile, *, reference: DapiRaster, frame: str, options: RasterOptions, max_instance_id: int
) -> DataArray | DataTree:
    """Return integer YX labels sharing the calibrated full-resolution DAPI grid."""
    metadata = _inspect(path, _SEGMENTATION_CONTEXT)
    labels = _array(metadata, _LABEL_AXES, _SEGMENTATION_CONTEXT)
    reference_shape = reference.data.shape[-2:]
    if labels.shape != reference_shape:
        raise ReaderFormatError(
            context=_SEGMENTATION_CONTEXT,
            path=path,
            found=f"segmentation YX shape {labels.shape} and DAPI shape {reference_shape}",
            expected="the same full-resolution pixel extent for segmentation and DAPI",
        )
    _check_calibration(metadata, reference.scale)
    if labels.dtype.kind not in {"i", "u"}:
        raise ReaderFormatError(
            context=_SEGMENTATION_CONTEXT,
            path=path,
            found=str(labels.dtype),
            expected="integer label pixels with zero reserved for background",
        )
    if max_instance_id > np.iinfo(labels.dtype).max:
        raise ReaderFormatError(
            context=_SEGMENTATION_CONTEXT,
            path=path,
            found="cell IDs exceeding the label dtype",
            expected=f"IDs representable by {labels.dtype}",
        )

    options.validate_label_pyramid(labels, element="the seqFISH segmentation")
    return Labels2DModel.parse(
        labels,
        dims=_LABEL_AXES,
        scale_factors=options.model_scale_factors,
        chunks=options.label_chunks(),
        transformations={frame: reference.scale},
    )
