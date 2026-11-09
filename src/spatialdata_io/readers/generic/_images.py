"""Read one caller-selected raster file as a lazy SpatialData image.

Supported files are selected by suffix:

- TIFF, OME-TIFF and BigTIFF (``.tif``, ``.tiff``, ``.btf``): the only series
  or an explicitly selected one, at any stored pyramid level, decoded lazily
  along the native tiles or strips.
- PNG (``.png``) and JPEG (``.jpg``, ``.jpeg``): one delayed whole-frame decode.

No axis meaning is guessed. The caller names every decoded axis with
``data_axes``, one string that is a permutation of ``yx``, ``cyx``, ``zyx`` or
``czyx``; the image is returned in canonical ``c, y, x`` or ``c, z, y, x``
order with a singleton channel added to grayscale data. Values, dtype and
channel order are kept. Axes that do not fit the decoded array raise
``ValueError`` with a note naming the image.

When the caller's ``c`` axis is the file's interleaved samples axis and the file
declares those samples RGB, the channels are ``r, g, b``, or ``r, g, b, a``
with a declared alpha sample. The declarations are TIFF
``PhotometricInterpretation`` RGB (alpha from ``ExtraSamples``) or
JPEG-compressed YCbCr, PNG colour types 2, 3 and 6 (alpha from colour type 6
or ``tRNS``), and three-component JPEG. Otherwise channels are named from
OME-XML only when the caller's ``c`` axis is the OME channel axis of the
selected series::

    declared  data_axes  declaration                       channel names
    YXS       yxc        Photometric RGB                   r, g, b
    YXS       yxc        RGB + ExtraSamples alpha          r, g, b, a
    YXS       yxc        PNG colour type 6                 r, g, b, a
    CYX       cyx        <Channel Name="DAPI"/> ...        DAPI, CD3, ...
    YXS       yxc        <Channel SamplesPerPixel="3"/>,   Channel:0:0:sample:0..2
                         Photometric MinIsBlack
    YX        yx         <Channel Name="DAPI"/>            DAPI
    ZYX       cyx        (records describe no Z planes)    0, 1, ... (positional)
    CYX       zyx        (channels read as Z)              0 (positional)

Names are the OME names when all are nonempty and unique, otherwise the OME
IDs. A reinterpreted axis, a samples axis next to a channel axis, and non-OME
sources without a colour declaration keep positional indexes. When the axes
do correspond, an OME channel count that disagrees with the raster or
duplicate IDs are corrupt metadata and raise.

Coordinates are pixels unless level zero of the selected series carries a
complete OME calibration of every retained spatial axis::

    <Pixels SizeX="24" SizeY="16" PhysicalSizeX="0.5" PhysicalSizeXUnit="µm"
            PhysicalSizeY="2.0" PhysicalSizeYUnit="µm" ...>

Then a ``Scale`` maps pixels to micrometres (µm, nm and mm are converted),
anisotropy included and without a half-pixel shift or stage origin. A stored
level other than zero, missing sizes, or ``data_axes`` that move a spatial axis
away from its declared position keep pixels, because the declared sizes no
longer describe the returned grid. OME schema:
https://www.openmicroscopy.org/Schemas/OME/2016-06
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from spatialdata.models import Image2DModel, Image3DModel
from spatialdata.transformations import Identity, Scale

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import (
    LazyRaster,
    channel_source,
    inspect_tiff,
    make_raster_options,
    normalize_raster_axes,
    ome_channel_names,
    ome_pixel_sizes_um,
    raster_channel_names,
    read_jpeg,
    read_png,
    read_tiff,
    select_tiff_series,
    validate_tiff_index,
)
from spatialdata_io.readers._utils.paths import require_file

if TYPE_CHECKING:
    from collections.abc import Sequence

    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import TiffMetadata

# sphinx-autodoc-typehints currently renders PEP 695 aliases as ``TypeAliasType``.
Chunks: TypeAlias = int | tuple[int, ...] | Mapping[str, int]  # noqa: UP040

type _ImageFormat = Literal["tiff", "png", "jpeg"]

_FORMATS: dict[str, _ImageFormat] = {
    ".btf": "tiff",
    ".tif": "tiff",
    ".tiff": "tiff",
    ".png": "png",
    ".jpeg": "jpeg",
    ".jpg": "jpeg",
}
VALID_IMAGE_TYPES = tuple(_FORMATS)

_CANONICAL_AXES: dict[frozenset[str], tuple[str, ...]] = {
    frozenset("yx"): ("c", "y", "x"),
    frozenset("cyx"): ("c", "y", "x"),
    frozenset("zyx"): ("c", "z", "y", "x"),
    frozenset("czyx"): ("c", "z", "y", "x"),
}
_SPATIAL_AXES = ("x", "y", "z")
_CONTEXT = ReaderErrorContext("generic", "image")


@dataclass(frozen=True, slots=True)
class _TiffSelection:
    """Inspected TIFF metadata with the resolved series and stored level."""

    metadata: TiffMetadata
    series: int
    level: int


def _image_format(path: Path) -> _ImageFormat:
    image_format = _FORMATS.get(path.suffix.lower())
    if image_format is None:
        message = f"Unsupported image suffix {path.suffix!r}. Expected one of {VALID_IMAGE_TYPES}."
        raise ValueError(message)
    return image_format


def _source_axes(data_axes: str) -> tuple[str, ...]:
    """Return the caller's lowercase axes, one string permuting a supported axis set."""
    if not isinstance(data_axes, str):
        message = f"data_axes must be a string such as 'cyx'; found {type(data_axes).__name__}."
        raise TypeError(message)
    axes = tuple(data_axes.lower())
    # The length check rejects repeated axes, which collapse in the set.
    if frozenset(axes) not in _CANONICAL_AXES or len(axes) != len(set(axes)):
        message = f"data_axes must be a permutation of 'yx', 'cyx', 'zyx', or 'czyx'; found {data_axes!r}."
        raise ValueError(message)
    return axes


def _validate_tiff_selection(image_format: _ImageFormat, *, tiff_series: int | None, tiff_level: int) -> None:
    validate_tiff_index(tiff_series, parameter="tiff_series", optional=True)
    validate_tiff_index(tiff_level, parameter="tiff_level", optional=False)
    if image_format != "tiff" and (tiff_series is not None or tiff_level != 0):
        message = "tiff_series and tiff_level can only select TIFF-family inputs."
        raise ValueError(message)


def _normalize_chunks(chunks: Chunks | None, *, axes: tuple[str, ...]) -> dict[str, int]:
    if chunks is None:
        # One channel and, for volumes, one plane with the shared y/x tiles.
        channel, tile_y, tile_x = DEFAULT_CHUNKS
        defaults = {"c": channel, "z": 1, "y": tile_y, "x": tile_x}
        return {axis: defaults[axis] for axis in axes}

    sizes: tuple[object, ...]
    if isinstance(chunks, Mapping):
        if set(chunks) != set(axes):
            message = f"chunks keys must match canonical axes {axes}. Found {tuple(chunks)}."
            raise ValueError(message)
        sizes = tuple(chunks[axis] for axis in axes)
    elif isinstance(chunks, tuple):
        if len(chunks) != len(axes):
            message = f"chunks length must match canonical axes {axes}. Found {chunks!r}."
            raise ValueError(message)
        sizes = chunks
    else:
        sizes = (chunks,) * len(axes)

    # Irregular per-axis partitions are rejected: the SpatialData writer would
    # silently regularize them to the largest chunk of each axis.
    integers = tuple(size for size in sizes if isinstance(size, int) and not isinstance(size, bool))
    if len(integers) != len(sizes):
        message = f"chunks must give one positive integer per canonical axis {axes}. Found {chunks!r}."
        raise TypeError(message)
    if min(integers) <= 0:
        message = f"chunks must be positive. Found {chunks!r}."
        raise ValueError(message)
    return dict(zip(axes, integers, strict=True))


def _read_raster(
    path: Path, image_format: _ImageFormat, *, tiff_series: int | None, tiff_level: int
) -> tuple[LazyRaster, _TiffSelection | None]:
    file = require_file(path, context=_CONTEXT)
    if image_format == "tiff":
        metadata = inspect_tiff(file, context=_CONTEXT)
        series = select_tiff_series(metadata, tiff_series, context=_CONTEXT, option="tiff_series")
        raster = read_tiff(metadata, series=series, level=tiff_level, context=_CONTEXT)
        return raster, _TiffSelection(metadata, series, tiff_level)
    if image_format == "png":
        return read_png(file, context=_CONTEXT), None
    return read_jpeg(file, context=_CONTEXT), None


def _spatial_positions(axes: tuple[str, ...]) -> tuple[str | None, ...]:
    return tuple(axis if axis in _SPATIAL_AXES else None for axis in axes)


def _physical_transform(raster: LazyRaster, tiff: _TiffSelection | None, axes: tuple[str, ...]) -> Identity | Scale:
    """Calibrate retained spatial axes from complete level-zero OME sizes, otherwise keep pixels."""
    if tiff is None or tiff.level != 0:
        return Identity()
    metadata, series = tiff.metadata, tiff.series
    if metadata.ome is None:
        return Identity()
    # An explicit reinterpretation of spatial axes breaks their OME association.
    declared = tuple(axis.lower() for axis in raster.axes)
    if _spatial_positions(declared) != _spatial_positions(axes):
        return Identity()

    pixels = metadata.ome.images[series].pixels
    dimensions = {"x": pixels.size_x, "y": pixels.size_y, "z": pixels.size_z}
    spatial_axes: tuple[Literal["x", "y", "z"], ...] = tuple(axis for axis in _SPATIAL_AXES if axis in axes)
    for axis in spatial_axes:
        if dimensions[axis] != raster.data.shape[declared.index(axis)]:
            raise ReaderFormatError(
                context=_CONTEXT,
                path=metadata.path,
                found=f"OME {axis.upper()} dimension {dimensions[axis]} disagrees with its base raster",
                expected="matching source and OME spatial dimensions",
            )

    declared_sizes = ome_pixel_sizes_um(metadata, series=series, axes=spatial_axes, context=_CONTEXT)
    sizes = [size for size in declared_sizes if size is not None]
    # Mixing physical and pixel axes in one frame is not a valid calibration.
    return Scale(sizes, axes=spatial_axes) if len(sizes) == len(spatial_axes) else Identity()


def _channel_names(
    raster: LazyRaster, tiff: _TiffSelection | None, axes: tuple[str, ...], count: int
) -> list[str] | None:
    """Return declared colour or OME channel names, or ``None`` for positional indexes."""
    # The caller names interleaved samples "c" too; a colour declaration
    # concerns the samples alone, even next to a channel axis.
    samples, ome_samples = channel_source(tuple(axis.lower() for axis in raster.axes), axes)
    ome = None
    if tiff is not None and ome_samples is not None:
        # Count or identity disagreements on a corresponding axis are corrupt OME and raise.
        ome = ome_channel_names(tiff.metadata, series=tiff.series, count=count, samples=ome_samples, context=_CONTEXT)
    return raster_channel_names(raster.color, samples=samples, ome=ome)


# Explicit public options keep format and model contracts inspectable.
def image(  # noqa: PLR0913, RUF100
    path: str | Path,
    data_axes: str,
    coordinate_system: str,
    *,
    tiff_series: int | None = None,
    tiff_level: int = 0,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: Chunks | None = None,
) -> DataArray | DataTree:
    """Read a TIFF, PNG or JPEG file as a lazy SpatialData image.

    Supported inputs are TIFF, OME-TIFF and BigTIFF (``.tif``, ``.tiff``,
    ``.btf``), PNG (``.png``) and JPEG (``.jpg``, ``.jpeg``), selected by the
    case-insensitive suffix.

    Parameters
    ----------
    path
        Raster file to read. ``~`` is expanded.
    data_axes
        Caller-declared axes of the decoded array as one string, for example
        ``"yxc"``: a case-insensitive permutation of ``yx``, ``cyx``, ``zyx``
        or ``czyx``. Interleaved TIFF, PNG and JPEG samples (for example RGB)
        are named ``c`` here.
    coordinate_system
        Name of the target coordinate system, preserved literally; it must
        contain a non-whitespace character.
    tiff_series
        Zero-based TIFF series. ``None`` reads the only series and fails for a
        file with several. Must be ``None`` for PNG and JPEG.
    tiff_level
        Zero-based stored pyramid level of the selected series. Must be zero
        for PNG and JPEG.
    scale_factors
        Integer factors of at least two, one per pyramid level generated from
        the selected source level; empty for a single scale.
    chunks
        One positive size per canonical axis, applied at every pyramid level:
        an integer for every axis, a tuple in canonical ``CYX``/``CZYX`` order,
        or a mapping keyed by exactly the canonical axes. ``None`` selects one
        channel and one plane per chunk with square 1024-pixel tiles:
        ``(1, 1024, 1024)`` or ``(1, 1, 1024, 1024)``.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        A 2D ``(c, y, x)`` or 3D ``(c, z, y, x)`` image, multiscale when
        ``scale_factors`` is not empty.

        - Values, dtype and channel order are those of the source; grayscale
          data receives a singleton channel.
        - Channel coordinates: ``r, g, b`` (``r, g, b, a`` with declared
          alpha) when the caller's ``c`` axis is the file's samples axis and
          the file declares them RGB: TIFF ``PhotometricInterpretation`` RGB
          or JPEG-compressed YCbCr, PNG colour types 2, 3 and 6, and
          three-component JPEG. Otherwise OME channel names (or IDs when
          names are missing or repeated) when the caller's ``c`` axis is the
          declared OME channel axis, where the samples of one OME channel are
          named ``<name or ID>:sample:<k>``. Otherwise, including non-OME
          TIFFs, grayscale PNG and reinterpreted channel axes, the positional
          indexes ``0, 1, ...``.
        - Coordinates: complete level-zero OME physical sizes of every retained
          spatial axis give a ``Scale`` into micrometres. Otherwise, including
          nonzero stored levels, partial calibration and reinterpreted spatial
          axes, an ``Identity`` keeps source pixels. No origin is inferred.
        - Laziness: construction decodes no pixels. A computed TIFF region
          decodes the source tiles or strips its chunks intersect; a PNG or
          JPEG computation decodes the whole frame.

    Raises
    ------
    TypeError
        If ``data_axes`` or ``coordinate_system`` is not a string, or the TIFF
        selection, scale factors or chunk sizes have the wrong type, including
        irregular per-axis chunk partitions.
    ValueError
        If the suffix, axes, coordinate system name, TIFF selection, scale
        factors or chunks are invalid, or the axes do not match the decoded
        array (with a note naming the image). Malformed files raise the
        subclasses ``ReaderFormatError`` or ``RasterFormatError`` with the
        reader context, including a TIFF with several series and no
        ``tiff_series``, and OME channel records that disagree with the raster
        channels.
    IndexError
        If the TIFF series or level does not exist.
    FileNotFoundError, IsADirectoryError
        If ``path`` does not name an existing regular file.

    Notes
    -----
    TIFF metadata inspection is eager and bounded. Caller-owned sequences and
    mappings are not mutated.
    """
    source = Path(path).expanduser()
    image_format = _image_format(source)
    source_axes = _source_axes(data_axes)
    target_axes = _CANONICAL_AXES[frozenset(source_axes)]
    if not isinstance(coordinate_system, str):
        message = "coordinate_system must be a string."
        raise TypeError(message)
    if not coordinate_system.strip():
        message = "coordinate_system must contain a non-whitespace character."
        raise ValueError(message)
    _validate_tiff_selection(image_format, tiff_series=tiff_series, tiff_level=tiff_level)
    # Generic chunks are axis-aware and validated separately; the shared options own the factors.
    options = make_raster_options(scale_factors)
    chunks_by_axis = _normalize_chunks(chunks, axes=target_axes)

    raster, tiff = _read_raster(source, image_format, tiff_series=tiff_series, tiff_level=tiff_level)
    try:
        normalized = normalize_raster_axes(
            raster.data,
            source_axes,
            target_axes=target_axes,
            add_missing=("c",) if "c" not in source_axes else (),
        )
    except ValueError as error:
        # The axes are the caller's, so a mismatch with the file is a caller error.
        add_reader_context(error, context=_CONTEXT, path=source)
        raise
    options.validate_image_pyramid(normalized, element="the image")
    transform = _physical_transform(raster, tiff, source_axes)
    channel_names = _channel_names(raster, tiff, source_axes, int(normalized.shape[0]))

    model = Image2DModel if target_axes == ("c", "y", "x") else Image3DModel
    return model.parse(
        normalized,
        dims=target_axes,
        c_coords=channel_names,
        transformations={coordinate_system: transform},
        scale_factors=options.model_scale_factors,
        chunks=chunks_by_axis,
    )
