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
                   either declaration

RGB(A) is declared by TIFF ``PhotometricInterpretation`` RGB (alpha through
``ExtraSamples``) or JPEG-compressed YCbCr, PNG colour types 2, 3 and 6 (alpha
through colour type 6 or ``tRNS``), and three-component JPEG. OME declares an
interleaved image as one ``Channel`` with several samples, so samples the file
does not declare RGB are named ``<name or ID>:sample:<index>``. The channel
count or dtype never makes an image RGB.

A TIFF has one series unless ``tiff_series`` selects one of several;
``tiff_level`` selects a stored pyramid level. The image maps by identity to
``image_pixels``; the reader scales the grid into that frame using the
selected level's full-resolution extent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    normalize_raster_axes,
    ome_channel_names,
    raster_channel_names,
    read_jpeg,
    read_png,
    read_tiff,
)
from spatialdata_io.readers.dbit._schemas import IMAGE_FRAME

if TYPE_CHECKING:
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions, TiffMetadata
    from spatialdata_io.readers._utils.paths import ArtifactFile

_IMAGE_AXES = ("c", "y", "x")
_TIFF_SUFFIXES = frozenset({".tif", ".tiff", ".btf"})
_CONTEXT = ReaderErrorContext("dbit", "histology image")


def validate_image_options(data_axes: str | None, tiff_series: int | None, tiff_level: int) -> None:
    """Validate the caller's source axes and TIFF selections before any I/O.

    The rank of ``data_axes`` and the range of the TIFF indexes depend on the
    file and are checked when it is read.
    """
    if data_axes is not None:
        if not isinstance(data_axes, str):
            message = "data_axes must be a string such as 'CYX', or None."
            raise TypeError(message)
        # S (samples) becomes the channel axis, so it may not accompany C.
        axes = data_axes.lower().replace("s", "c")
        if not axes.isascii() or not axes.isalpha() or len(set(axes)) != len(axes) or not {"x", "y"}.issubset(axes):
            message = (
                "data_axes must contain distinct single-letter axes including X and Y, "
                f"with S (samples) counting as C; found {data_axes!r}."
            )
            raise ValueError(message)

    if tiff_series is not None:
        _validate_index(tiff_series, name="tiff_series")
    _validate_index(tiff_level, name="tiff_level")


def _validate_index(value: int, *, name: str) -> None:
    """Require a nonnegative integer that is not a boolean."""
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"{name} must be an integer."
        raise TypeError(message)
    if value < 0:
        message = f"{name} must be nonnegative; found {value}."
        raise ValueError(message)


def _only_series(metadata: TiffMetadata, path: ArtifactFile) -> int:
    """Return the index of the file's only series; several require ``tiff_series``."""
    if len(metadata.series) != 1:
        raise ReaderFormatError(
            context=_CONTEXT,
            path=path,
            found=f"{len(metadata.series)} series",
            expected="a single series or an explicit tiff_series selection",
        )
    return 0


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

    metadata, series = None, 0
    if is_tiff:
        metadata = inspect_tiff(path, context=_CONTEXT)
        series = _only_series(metadata, path) if tiff_series is None else tiff_series
        lazy = read_tiff(metadata, series=series, level=tiff_level, context=_CONTEXT)
    elif path.suffix.lower() == ".png":
        lazy = read_png(path, context=_CONTEXT)
    else:
        lazy = read_jpeg(path, context=_CONTEXT)

    declared = tuple(axis.lower() for axis in lazy.axes)
    source = declared if data_axes is None else tuple(data_axes.lower())
    try:
        image = normalize_raster_axes(
            lazy.data,
            tuple("c" if axis == "s" else axis for axis in source),
            target_axes=_IMAGE_AXES,
            add_missing=("c",),
        )
    except ValueError as error:
        add_reader_context(error, context=_CONTEXT, path=path)
        raise
    raster.validate_image_pyramid(image, element="the histology image")

    # Declared RGB samples and OME sample records both apply only when the
    # file's samples axis became the channels.
    samples = "s" in declared and source[declared.index("s")] in {"c", "s"}
    ome = None
    if metadata is not None:
        ome = ome_channel_names(metadata, series=series, count=int(image.shape[0]), samples=samples, context=_CONTEXT)

    return Image2DModel.parse(
        image,
        dims=_IMAGE_AXES,
        c_coords=raster_channel_names(lazy.color, samples=samples, ome=ome),
        rgb=False,
        scale_factors=raster.model_scale_factors,
        chunks=raster.image_chunks(),
        transformations={IMAGE_FRAME: Identity()},
    )
