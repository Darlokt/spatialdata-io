"""Read the optional DBiT tissue image lazily as a CYX image element.

DBiT exports ship a low-resolution tissue image (``tissue_lowres_image.png``)
that is already cropped to the barcode grid and registered to it; a caller can
supply any other TIFF, OME-TIFF, BigTIFF/BTF, PNG or JPEG file that meets the
same assumption. The reader does not register, crop or orient the image::

    source axes    TIFF: file-declared (``YX``, ``CYX``, ``YXS``, ...) or ``data_axes``;
                   PNG and JPEG: ``YXS``
    samples (S)    interleaved samples become channels
    other axes     singleton dimensions are dropped; any other extra axis fails
    channels       ``r, g, b(, a)`` for samples the file declares RGB(A);
                   otherwise OME channel names when all are nonempty and
                   unique, else OME channel IDs; positional integers without
                   either declaration or when ``data_axes`` reads another axis
                   as the channels

RGB(A) is declared by TIFF ``PhotometricInterpretation`` RGB (alpha through
``ExtraSamples``) or JPEG-compressed YCbCr, PNG colour types 2, 3 and 6 (alpha
through colour type 6 or ``tRNS``), and three-component JPEG. OME declares an
interleaved image as one ``Channel`` with several samples, so samples the file
does not declare RGB are named ``<name or ID>:sample:<index>``. The channel
count or dtype never makes an image RGB.

Axes the file declares that cannot become ``c, y, x`` are malformed content and
raise ``ReaderFormatError``; axes from ``data_axes`` that do not fit the file
raise ``ValueError`` with a note naming the image.

A TIFF has one series unless ``tiff_series`` selects one of several;
``tiff_level`` selects a stored pyramid level. The image maps by identity to
``image_pixels``; the reader scales the grid into that frame using the
selected level's full-resolution extent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    normalize_cyx_image,
    read_jpeg,
    read_png,
    read_tiff,
    select_tiff_series,
    validate_data_axes,
    validate_tiff_index,
)
from spatialdata_io.readers.dbit._schemas import IMAGE_FRAME

if TYPE_CHECKING:
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers._utils.paths import ArtifactFile

_IMAGE_AXES = ("c", "y", "x")
_TIFF_SUFFIXES = frozenset({".tif", ".tiff", ".btf"})
_CONTEXT = ReaderErrorContext("dbit", "histology image")


def validate_image_options(data_axes: str | None, tiff_series: int | None, tiff_level: int) -> None:
    """Validate the caller's source axes and TIFF selections before any I/O.

    The rank of ``data_axes`` and the range of the TIFF indexes depend on the
    file and are checked when it is read.
    """
    validate_data_axes(data_axes)
    validate_tiff_index(tiff_series, parameter="tiff_series", optional=True)
    validate_tiff_index(tiff_level, parameter="tiff_level", optional=False)


def read_image(
    path: ArtifactFile,
    *,
    data_axes: str | None,
    tiff_series: int | None,
    tiff_level: int,
    raster: RasterOptions,
) -> DataArray | DataTree:
    """Read bounded metadata and build the image without executing pixel tasks."""
    is_tiff = path.suffix.lower() in _TIFF_SUFFIXES
    if not is_tiff and (data_axes is not None or tiff_series is not None or tiff_level != 0):
        message = "data_axes, tiff_series and tiff_level require a TIFF-family image."
        raise ValueError(message)

    tiff = None
    if is_tiff:
        metadata = inspect_tiff(path, context=_CONTEXT)
        series = select_tiff_series(metadata, tiff_series, context=_CONTEXT, option="tiff_series")
        lazy = read_tiff(metadata, series=series, level=tiff_level, context=_CONTEXT)
        tiff = (metadata, series)
    elif path.suffix.lower() == ".png":
        lazy = read_png(path, context=_CONTEXT)
    else:
        lazy = read_jpeg(path, context=_CONTEXT)

    image, channels = normalize_cyx_image(lazy, data_axes, tiff=tiff, context=_CONTEXT, path=path)
    raster.validate_image_pyramid(image, element="the histology image")
    return Image2DModel.parse(
        image,
        dims=_IMAGE_AXES,
        c_coords=channels,
        rgb=False,
        scale_factors=raster.model_scale_factors,
        chunks=raster.image_chunks(),
        transformations={IMAGE_FRAME: Identity()},
    )
