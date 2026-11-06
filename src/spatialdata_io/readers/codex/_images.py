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
                   OME metadata

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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata.models import Image2DModel

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    normalize_raster_axes,
    ome_channel_names,
    raster_channel_names,
    read_tiff,
)

if TYPE_CHECKING:
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers._utils.paths import ArtifactFile

_IMAGE_AXES = ("c", "y", "x")
_IMAGE_CONTEXT = ReaderErrorContext("codex", "stitched image")
_CHANNELS_CONTEXT = ReaderErrorContext("codex", "stitched image channels")


def validate_image_options(data_axes: str | None, tiff_series: int | None) -> None:
    """Validate the caller's source axes and series selection before any I/O.

    The rank of ``data_axes`` and the range of ``tiff_series`` depend on the
    file and are checked when it is read.
    """
    if data_axes is not None:
        if not isinstance(data_axes, str):
            message = "data_axes must be a string such as 'CYX', or None."
            raise TypeError(message)
        if not data_axes.isalpha() or len(set(data_axes.lower())) != len(data_axes):
            message = f"data_axes must contain distinct single-letter axis names, such as 'CYX'; found {data_axes!r}."
            raise ValueError(message)

    if tiff_series is not None:
        if isinstance(tiff_series, bool) or not isinstance(tiff_series, int):
            message = "tiff_series must be an integer or None."
            raise TypeError(message)
        if tiff_series < 0:
            message = f"tiff_series must be nonnegative; found {tiff_series}."
            raise ValueError(message)


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
    if tiff_series is None:
        if len(metadata.series) != 1:
            raise ReaderFormatError(
                context=_IMAGE_CONTEXT,
                path=path,
                found=f"{len(metadata.series)} series",
                expected="an explicit tiff_series selection",
            )
        tiff_series = 0

    lazy = read_tiff(metadata, series=tiff_series, level=0, context=_IMAGE_CONTEXT)
    declared = tuple(axis.lower() for axis in lazy.axes)
    source = declared if data_axes is None else tuple(data_axes.lower())
    image = normalize_raster_axes(
        lazy.data, tuple("c" if axis == "s" else axis for axis in source), target_axes=_IMAGE_AXES, add_missing=("c",)
    )
    raster.validate_image_pyramid(image, element="the stitched image")

    # Declared RGB samples and OME sample records both apply only when the
    # file's samples axis became the channels.
    samples = "s" in declared and source[declared.index("s")] in {"c", "s"}
    ome = ome_channel_names(
        metadata, series=tiff_series, count=int(image.shape[0]), samples=samples, context=_CHANNELS_CONTEXT
    )
    channel_names = raster_channel_names(lazy.color, samples=samples, ome=ome)

    return Image2DModel.parse(
        image,
        dims=_IMAGE_AXES,
        c_coords=channel_names,
        scale_factors=raster.model_scale_factors,
        chunks=raster.image_chunks(),
        rgb=False,
        transformations={frame: transformation},
    )
