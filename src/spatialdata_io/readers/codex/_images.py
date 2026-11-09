"""Read one stitched CODEX TIFF-family image lazily as a CYX image element.

The processor writes stitched region images as TIFF, OME-TIFF, BigTIFF or BTF,
for example one multichannel stack per region. The reader selects one TIFF
series (the only one, or ``tiff_series``) and its full-resolution level::

    source axes    file-declared (``CYX``, ``TZCYX``, ``YXS``, ...) or ``data_axes``
    samples (S)    interleaved samples become channels
    other axes     singleton dimensions are dropped; any other extra axis fails
    channels       ``r, g, b(, a)`` for samples the TIFF declares RGB(A);
                   otherwise OME channel names when all are nonempty and
                   unique, else OME channel IDs; positional integers without
                   OME metadata or when ``data_axes`` reads another axis as
                   the channels

A TIFF declares RGB samples with ``PhotometricInterpretation`` RGB (or
JPEG-compressed YCbCr), and a fourth alpha sample with ``ExtraSamples``. OME
declares an interleaved image as one ``Channel`` with ``SamplesPerPixel="3"``;
when the channels come from samples the TIFF does not declare RGB, each sample
is named ``<name or ID>:sample:<index>``, for example ``Channel:0:0:sample:0``
to ``Channel:0:0:sample:2``.

Channels are not matched to table markers, and three- or four-channel stacks
are never guessed to be RGB from their count or dtype. The image keeps its own
pixel grid: it is not registered to the cell table unless the caller supplies
``image_to_table``.

Axes the file declares that cannot become ``c, y, x`` are malformed content and
raise ``ReaderFormatError``; axes from ``data_axes`` that do not fit the file
raise ``ValueError`` with a note naming the image.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata.models import Image2DModel

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    normalize_cyx_image,
    read_tiff,
    select_tiff_series,
    validate_data_axes,
    validate_tiff_index,
)

if TYPE_CHECKING:
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers._utils.paths import ArtifactFile

_IMAGE_AXES = ("c", "y", "x")
_IMAGE_CONTEXT = ReaderErrorContext("codex", "stitched image")


def validate_image_options(data_axes: str | None, tiff_series: int | None) -> None:
    """Validate the caller's source axes and series selection before any I/O.

    The rank of ``data_axes`` and the range of ``tiff_series`` depend on the
    file and are checked when it is read.
    """
    validate_data_axes(data_axes)
    validate_tiff_index(tiff_series, parameter="tiff_series", optional=True)


def read_image(  # noqa: PLR0913, RUF100 - Explicit raster and registration state.
    path: ArtifactFile,
    *,
    data_axes: str | None,
    tiff_series: int | None,
    raster: RasterOptions,
    frame: str,
    transformation: BaseTransformation,
) -> DataArray | DataTree:
    """Build a lazy CYX image from level zero of the selected TIFF series."""
    metadata = inspect_tiff(path, context=_IMAGE_CONTEXT)
    series = select_tiff_series(metadata, tiff_series, context=_IMAGE_CONTEXT, option="tiff_series")
    lazy = read_tiff(metadata, series=series, level=0, context=_IMAGE_CONTEXT)
    image, channel_names = normalize_cyx_image(
        lazy, data_axes, tiff=(metadata, series), context=_IMAGE_CONTEXT, path=path
    )
    raster.validate_image_pyramid(image, element="the stitched image")
    return Image2DModel.parse(
        image,
        dims=_IMAGE_AXES,
        c_coords=channel_names,
        scale_factors=raster.model_scale_factors,
        chunks=raster.image_chunks(),
        rgb=False,
        transformations={frame: transformation},
    )
