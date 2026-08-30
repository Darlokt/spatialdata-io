"""Generic raster loading and SpatialData image-model construction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from spatialdata.models import Image2DModel, Image3DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.image import (
    LazyRaster,
    inspect_tiff,
    normalize_raster_axes,
    read_jpeg,
    read_png,
    read_tiff,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from xarray import DataArray, DataTree

# sphinx-autodoc-typehints currently renders PEP 695 aliases as ``TypeAliasType``.
ChunkValue: TypeAlias = int | tuple[int, ...] | None  # noqa: UP040
Chunks: TypeAlias = (  # noqa: UP040
    int | tuple[int, ...] | tuple[tuple[int, ...], ...] | Mapping[str, ChunkValue]
)

DEFAULT_SPATIAL_CHUNK_SIZE = 1024
TIFF_SUFFIXES = (".btf", ".tif", ".tiff")
PNG_SUFFIXES = (".png",)
JPEG_SUFFIXES = (".jpeg", ".jpg")
VALID_IMAGE_TYPES = TIFF_SUFFIXES + PNG_SUFFIXES + JPEG_SUFFIXES

_IMAGE_2D_AXES = frozenset(("c", "x", "y"))
_IMAGE_2D_GRAYSCALE_AXES = frozenset(("x", "y"))
_IMAGE_3D_AXES = frozenset(("c", "x", "y", "z"))
_IMAGE_3D_GRAYSCALE_AXES = frozenset(("x", "y", "z"))


def _canonical_axes(data_axes: Sequence[str]) -> tuple[str, ...]:
    axes = tuple(axis.lower() for axis in data_axes)
    if any(len(axis) != 1 or not axis.isalpha() for axis in axes):
        message = f"Invalid data_axes {tuple(data_axes)!r}; expected single alphabetic names."
        raise ValueError(message)
    if len(set(axes)) != len(axes):
        message = f"data_axes contains duplicate axes: {tuple(data_axes)!r}."
        raise ValueError(message)

    axis_set = frozenset(axes)
    if axis_set in (_IMAGE_2D_AXES, _IMAGE_2D_GRAYSCALE_AXES):
        return ("c", "y", "x")
    if axis_set in (_IMAGE_3D_AXES, _IMAGE_3D_GRAYSCALE_AXES):
        return ("c", "z", "y", "x")

    message = f"data_axes must be a permutation of 'yx', 'cyx', 'zyx', or 'czyx'; found {axes!r}."
    raise ValueError(message)


def _validate_tiff_selection(tiff_series: int, tiff_level: int) -> None:
    if isinstance(tiff_series, bool) or not isinstance(tiff_series, int) or tiff_series < 0:
        message = f"tiff_series must be a non-negative integer; found {tiff_series!r}."
        raise ValueError(message)
    if isinstance(tiff_level, bool) or not isinstance(tiff_level, int) or tiff_level < 0:
        message = f"tiff_level must be a non-negative integer; found {tiff_level!r}."
        raise ValueError(message)


def _validate_chunk_value(value: ChunkValue, *, axis: str) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        message = f"chunks for axis {axis!r} must not be a boolean."
        raise TypeError(message)
    if isinstance(value, int):
        if value <= 0:
            message = f"chunks for axis {axis!r} must be positive; found {value}."
            raise ValueError(message)
        return
    if not isinstance(value, tuple):
        message = f"chunks for axis {axis!r} must be an integer, tuple of integers, or None; found {value!r}."
        raise TypeError(message)
    if not value:
        message = f"chunks for axis {axis!r} must not be empty."
        raise ValueError(message)
    if any(isinstance(size, bool) or not isinstance(size, int) for size in value):
        message = f"chunks for axis {axis!r} must contain only integers; found {value!r}."
        raise TypeError(message)
    if any(size <= 0 for size in value):
        message = f"chunks for axis {axis!r} must contain only positive sizes; found {value!r}."
        raise ValueError(message)


def _normalize_chunks(chunks: Chunks | None, *, axes: tuple[str, ...]) -> dict[str, ChunkValue]:
    if chunks is None:
        return {axis: DEFAULT_SPATIAL_CHUNK_SIZE if axis in {"x", "y"} else None for axis in axes}
    if isinstance(chunks, bool):
        message = "chunks must be a chunk specification, not a boolean."
        raise TypeError(message)
    if isinstance(chunks, int):
        normalized: dict[str, ChunkValue] = dict.fromkeys(axes, chunks)
    elif isinstance(chunks, Mapping):
        if set(chunks) != set(axes):
            message = f"chunks keys must match canonical axes {axes}; found {tuple(chunks)}."
            raise ValueError(message)
        normalized = {axis: chunks[axis] for axis in axes}
    elif isinstance(chunks, tuple):
        if len(chunks) != len(axes):
            message = f"chunks length must match canonical axes {axes}; found {chunks!r}."
            raise ValueError(message)
        if chunks and not (
            all(isinstance(value, int) for value in chunks) or all(isinstance(value, tuple) for value in chunks)
        ):
            message = "chunks must contain either one integer or one tuple of integers per canonical axis."
            raise TypeError(message)
        normalized = dict(zip(axes, chunks, strict=True))
    else:
        message = f"chunks must be an integer, tuple, mapping, or None; found {type(chunks).__name__}."
        raise TypeError(message)

    for axis, value in normalized.items():
        _validate_chunk_value(value, axis=axis)
    return normalized


def _read_raster(path: Path, *, tiff_series: int, tiff_level: int) -> LazyRaster:
    suffix = path.suffix.lower()
    if suffix in TIFF_SUFFIXES:
        metadata = inspect_tiff(path)
        return read_tiff(metadata, series=tiff_series, level=tiff_level)

    if tiff_series != 0 or tiff_level != 0:
        message = "tiff_series and tiff_level can only select TIFF-family inputs."
        raise ValueError(message)
    if suffix in PNG_SUFFIXES:
        return read_png(path)
    if suffix in JPEG_SUFFIXES:
        return read_jpeg(path)

    message = f"Unsupported image suffix {path.suffix!r}; expected one of {VALID_IMAGE_TYPES}."
    raise ValueError(message)


# Explicit public options keep format and model contracts inspectable.
def image(  # noqa: PLR0913, RUF100
    path: str | Path,
    data_axes: Sequence[str],
    coordinate_system: str,
    *,
    tiff_series: int = 0,
    tiff_level: int = 0,
    chunks: Chunks | None = None,
    scale_factors: Sequence[int] | None = None,
) -> DataArray | DataTree:
    """Read a generic raster as a lazy SpatialData image.

    TIFF, BigTIFF/BTF, PNG, and JPEG inputs are supported. ``data_axes`` is
    the caller's explicit interpretation of the decoded array and must be a
    permutation of ``YX``, ``CYX``, ``ZYX``, or ``CZYX``. Grayscale inputs
    receive a singleton channel axis. Returned images use canonical ``CYX`` or
    ``CZYX`` dimensions without decoding pixels during construction.

    TIFF-family sources retain their native storage-aligned source graph;
    requested model chunks add only lazy rechunking. PNG and JPEG decoding is
    delayed but necessarily whole-frame.

    Parameters
    ----------
    path
        Raster file to read.
    data_axes
        Explicit axes of the decoded source array.
    coordinate_system
        Coordinate system receiving an identity transformation.
    tiff_series
        Zero-based TIFF series. Must remain zero for PNG and JPEG inputs.
    tiff_level
        Zero-based pyramid level within the selected TIFF series. Must remain
        zero for PNG and JPEG inputs.
    chunks
        Requested chunks in canonical output-axis order or by canonical axis
        name. By default, only the spatial axes are rechunked to 1024; channel
        and depth axes retain the source chunking. An integer or positional
        tuple explicitly chunks every output axis. A mapping can use ``None``
        to retain the source chunks for selected axes.
    scale_factors
        Relative factors used to construct a model pyramid from the selected
        source level.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        A single-scale or multiscale 2D/3D SpatialData image.

    Raises
    ------
    TypeError
        If a chunk specification contains unsupported value types.
    ValueError
        If axes, TIFF selection, chunks, or the suffix are invalid.

    Notes
    -----
    This function does not mutate caller-owned mappings or arrays. TIFF
    metadata inspection is eager and bounded; raster pixel decoding is lazy.
    """
    _validate_tiff_selection(tiff_series, tiff_level)
    target_axes = _canonical_axes(data_axes)
    chunks_by_axis = _normalize_chunks(chunks, axes=target_axes)

    raster = _read_raster(Path(path), tiff_series=tiff_series, tiff_level=tiff_level)
    normalized = normalize_raster_axes(
        raster.data,
        data_axes,
        target_axes=target_axes,
        add_missing=("c",) if "c" not in {axis.lower() for axis in data_axes} else (),
    )
    model = Image2DModel if target_axes == ("c", "y", "x") else Image3DModel
    return model.parse(
        normalized,
        dims=target_axes,
        transformations={coordinate_system: Identity()},
        scale_factors=scale_factors,
        chunks=chunks_by_axis,
    )
