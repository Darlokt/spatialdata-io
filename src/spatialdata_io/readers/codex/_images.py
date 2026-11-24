"""Read flat TIFF stacks, Processor marker planes, and Uploader montages.

For each source the reader selects the full-resolution level of the only
series or the caller's ``tiff_series``. All outputs are lazy ``c, y, x``.

Flat TIFF, OME-TIFF, BigTIFF or BTF stacks use file-declared axes or an
explicit ``data_axes`` override. Interleaved samples become channels, and
other axes must be singleton. Processor marker files each contribute one
YX plane of the same shape and dtype, assembled in numeric cycle/channel
order. The marker name comes from each filename.

Nolan Uploader montages use ImageJ hyperstacks with cycles as frames (T),
acquisition channels as C, and one best-focus Z plane. The reader flattens
T and C in cycle-major order, preserving the acquisition-channel order
within each cycle. Complete ImageJ labels have priority over the nearest
``channelNames.txt``, followed by generated cycle/channel identifiers.
The sidecar holds one nonblank name per plane and is limited to 1 MiB.

A flat TIFF declares RGB samples with ``PhotometricInterpretation`` RGB (or
JPEG-compressed YCbCr), and a fourth alpha sample with ``ExtraSamples``. OME
declares an interleaved image as one ``Channel`` with ``SamplesPerPixel="3"``;
when the channels come from samples the TIFF does not declare RGB, each sample
is named ``<name or ID>:sample:<index>``, for example ``Channel:0:0:sample:0``
to ``Channel:0:0:sample:2``. Otherwise flat channels use unique OME names,
OME IDs, or positional indexes.

An OME scale requires both physical XY sizes for the stored axes. Marker
planes must agree wherever multiple files declare the same scale. Each image
retains its own grid and channel order. The caller supplies ``image_to_table``
to register the raster to a cell table, including a downsampled montage.

Axes the file declares that cannot be normalized raise ``ReaderFormatError``;
axes from ``data_axes`` that do not fit the file
raise ``ValueError`` with a note naming the image.
"""

from __future__ import annotations

import codecs
from typing import TYPE_CHECKING

import dask.array as da
import pandas as pd
from anndata.utils import make_index_unique
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    normalize_cyx_image,
    normalize_raster_axes,
    ome_pixel_sizes_um,
    read_tiff,
    require_matching_pixel_sizes,
    select_tiff_series,
    validate_data_axes,
    validate_tiff_index,
)
from spatialdata_io.readers.codex._layout import MarkerSet, Montage

if TYPE_CHECKING:
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers._utils.image.tiff import TiffMetadata
    from spatialdata_io.readers._utils.paths import ArtifactFile

_IMAGE_AXES = ("c", "y", "x")
_IMAGE_CONTEXT = ReaderErrorContext("codex", "stitched image")
_CHANNEL_CONTEXT = ReaderErrorContext("codex", "montage channel names")
_MAX_CHANNEL_NAMES_BYTES = 1024 * 1024


def validate_image_options(data_axes: str | None, tiff_series: int | None) -> None:
    """Validate the caller's source axes and series selection before any I/O.

    The rank of ``data_axes`` and the range of ``tiff_series`` depend on the
    file and are checked when it is read.
    """
    validate_data_axes(data_axes)
    validate_tiff_index(tiff_series, parameter="tiff_series", optional=True)


def _montage_names(path: ArtifactFile, count: int) -> list[str]:
    try:
        with path.open("rb") as handle:
            data = handle.read(_MAX_CHANNEL_NAMES_BYTES + 1)
    except OSError as error:
        add_reader_context(error, context=_CHANNEL_CONTEXT, path=path)
        raise
    if len(data) > _MAX_CHANNEL_NAMES_BYTES:
        raise ReaderFormatError(
            context=_CHANNEL_CONTEXT,
            path=path,
            found="more than 1 MiB",
            expected="a channel-name file of at most 1 MiB",
        )
    encoding = "utf-16" if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else "utf-8-sig"
    try:
        names = [name.strip() for name in data.decode(encoding).splitlines()]
    except UnicodeDecodeError as error:
        raise ReaderFormatError(
            context=_CHANNEL_CONTEXT, path=path, found=str(error), expected="UTF-8 or BOM-marked UTF-16 text"
        ) from error
    if len(names) != count or any(not name or "\x00" in name for name in names):
        raise ReaderFormatError(
            context=_CHANNEL_CONTEXT,
            path=path,
            found=f"{len(names)} entries, possibly blank or containing NUL",
            expected=f"{count} nonblank channel names in acquisition order",
        )
    return names


def _montage_axes(image: da.Array, source: tuple[str, ...], path: ArtifactFile, *, overridden: bool) -> da.Array:
    try:
        return normalize_raster_axes(image, source, target_axes=("t", "c", "y", "x"), add_missing=("t", "c"))
    except ValueError as error:
        if overridden:
            add_reader_context(error, context=_IMAGE_CONTEXT, path=path)
            raise
        raise ReaderFormatError(
            context=_IMAGE_CONTEXT,
            path=path,
            found=str(error),
            expected="an Uploader montage with declared cycle (T), channel (C), and YX axes",
        ) from error


def _read_montage(
    source: Montage, data_axes: str | None, tiff_series: int | None
) -> tuple[da.Array, list[str], tuple[float | None, ...]]:
    path = source.path
    metadata = inspect_tiff(path, context=_IMAGE_CONTEXT, include_imagej_labels=True)
    series = select_tiff_series(metadata, tiff_series, context=_IMAGE_CONTEXT, option="tiff_series")
    raster = read_tiff(metadata, series=series, level=0, context=_IMAGE_CONTEXT)
    axes = tuple(axis.lower() for axis in raster.axes) if data_axes is None else tuple(data_axes.lower())
    # MakeMontage writes frames as cycles and interleaves channels within each frame.
    stack = _montage_axes(raster.data, axes, path, overridden=data_axes is not None)
    cycles, channels, height, width = map(int, stack.shape)
    count = cycles * channels
    labels = metadata.imagej_labels
    if labels is not None and len(labels) == count and all(label.strip() for label in labels):
        if "t" in axes and "c" in axes and axes.index("c") < axes.index("t"):
            names = [labels[channel * cycles + cycle] for cycle in range(cycles) for channel in range(channels)]
        else:
            names = list(labels)
    elif source.channel_names is not None:
        names = _montage_names(source.channel_names, count)
    else:
        names = [
            f"cyc{cycle:03d}_ch{channel:03d}" for cycle in range(1, cycles + 1) for channel in range(1, channels + 1)
        ]
    sizes = _pixel_sizes(metadata, series, raster.axes, data_axes)
    return da.reshape(stack, (count, height, width)), make_index_unique(pd.Index(names)).tolist(), sizes


def _read_tiff_image(
    path: ArtifactFile, data_axes: str | None, tiff_series: int | None
) -> tuple[da.Array, list[str] | None, tuple[float | None, ...]]:
    metadata = inspect_tiff(path, context=_IMAGE_CONTEXT)
    series = select_tiff_series(metadata, tiff_series, context=_IMAGE_CONTEXT, option="tiff_series")
    lazy = read_tiff(metadata, series=series, level=0, context=_IMAGE_CONTEXT)
    image, names = normalize_cyx_image(lazy, data_axes, tiff=(metadata, series), context=_IMAGE_CONTEXT, path=path)
    return image, names, _pixel_sizes(metadata, series, lazy.axes, data_axes)


def _pixel_sizes(
    metadata: TiffMetadata, series: int, source: tuple[str, ...], override: str | None
) -> tuple[float | None, ...]:
    # An override moving Y or X changes the association of metadata to axes.
    if override is not None and any(source[override.lower().index(axis)].lower() != axis for axis in ("x", "y")):
        return None, None
    return ome_pixel_sizes_um(metadata, series=series, axes=("x", "y"), context=_IMAGE_CONTEXT)


def _read_marker_set(
    markers: MarkerSet, data_axes: str | None, tiff_series: int | None
) -> tuple[da.Array, list[str], tuple[float | None, ...]]:
    pixels: list[da.Array] = []
    sizes: tuple[float | None, ...] = (None, None)
    for source in markers.planes:
        image, _, declared = _read_tiff_image(source.path, data_axes, tiff_series)
        if image.shape[0] != 1 or (pixels and (image.shape[1:] != pixels[0].shape or image.dtype != pixels[0].dtype)):
            raise ReaderFormatError(
                context=_IMAGE_CONTEXT,
                path=source.path,
                found=f"marker array of shape {image.shape} and dtype {image.dtype}",
                expected="one plane per marker, all sharing the same YX shape and dtype",
            )
        require_matching_pixel_sizes(
            declared,
            sizes,
            axes="xy",
            context=_IMAGE_CONTEXT,
            path=source.path,
            expected="identical declared pixel sizes for all marker planes",
        )
        sizes = tuple(
            existing if existing is not None else size for existing, size in zip(sizes, declared, strict=True)
        )
        pixels.append(image[0])
    names = make_index_unique(pd.Index([source.marker for source in markers.planes])).tolist()
    return da.stack(pixels), names, sizes


def read_image(  # noqa: PLR0913, RUF100 - Explicit raster and registration state.
    source: ArtifactFile | MarkerSet | Montage,
    *,
    data_axes: str | None,
    tiff_series: int | None,
    raster: RasterOptions,
    frame: str,
    transformation: BaseTransformation,
    physical_frame: str | None = None,
) -> DataArray | DataTree:
    """Build a lazy CYX image and apply its verifiable independent OME scale, when eligible."""
    if isinstance(source, MarkerSet):
        image, channel_names, sizes = _read_marker_set(source, data_axes, tiff_series)
    elif isinstance(source, Montage):
        image, channel_names, sizes = _read_montage(source, data_axes, tiff_series)
    else:
        image, channel_names, sizes = _read_tiff_image(source, data_axes, tiff_series)
    x, y = sizes
    if physical_frame is not None and x is not None and y is not None:
        frame, transformation = physical_frame, Scale([x, y], axes=("x", "y"))
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
