"""Generic raster loading and SpatialData image-model construction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from spatialdata.models import Image2DModel, Image3DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.image import (
    DEFAULT_CHUNKS,
    DEFAULT_SCALE_FACTORS,
    LazyRaster,
    inspect_tiff,
    make_raster_options,
    normalize_raster_axes,
    read_jpeg,
    read_png,
    read_tiff,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from xarray import DataArray, DataTree

# sphinx-autodoc-typehints currently renders PEP 695 aliases as ``TypeAliasType``.
Chunks: TypeAlias = int | tuple[int, ...] | Mapping[str, int]  # noqa: UP040

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
        message = f"Invalid data_axes {tuple(data_axes)!r}. Expected single alphabetic names."
        raise ValueError(message)
    if len(set(axes)) != len(axes):
        message = f"data_axes contains duplicate axes: {tuple(data_axes)!r}."
        raise ValueError(message)

    axis_set = frozenset(axes)
    if axis_set in (_IMAGE_2D_AXES, _IMAGE_2D_GRAYSCALE_AXES):
        return ("c", "y", "x")
    if axis_set in (_IMAGE_3D_AXES, _IMAGE_3D_GRAYSCALE_AXES):
        return ("c", "z", "y", "x")

    message = f"data_axes must be a permutation of 'yx', 'cyx', 'zyx', or 'czyx'. Found {axes!r}."
    raise ValueError(message)


def _validate_tiff_selection(tiff_series: int, tiff_level: int) -> None:
    if isinstance(tiff_series, bool) or not isinstance(tiff_series, int) or tiff_series < 0:
        message = f"tiff_series must be a non-negative integer. Found {tiff_series!r}."
        raise ValueError(message)
    if isinstance(tiff_level, bool) or not isinstance(tiff_level, int) or tiff_level < 0:
        message = f"tiff_level must be a non-negative integer. Found {tiff_level!r}."
        raise ValueError(message)


def _default_chunks(axes: tuple[str, ...]) -> dict[str, int]:
    # One channel and, for volumes, one plane per chunk with the shared y/x tiles.
    channel, tile_y, tile_x = DEFAULT_CHUNKS
    sizes = {"c": channel, "z": 1, "y": tile_y, "x": tile_x}
    return {axis: sizes[axis] for axis in axes}


def _normalize_chunks(chunks: Chunks | None, *, axes: tuple[str, ...]) -> dict[str, int]:
    if chunks is None:
        return _default_chunks(axes)

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

    message = f"Unsupported image suffix {path.suffix!r}. Expected one of {VALID_IMAGE_TYPES}."
    raise ValueError(message)


# Explicit public options keep format and model contracts inspectable.
def image(  # noqa: PLR0913, RUF100
    path: str | Path,
    data_axes: Sequence[str],
    coordinate_system: str,
    *,
    tiff_series: int = 0,
    tiff_level: int = 0,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: Chunks | None = None,
) -> DataArray | DataTree:
    """Read a generic raster as a lazy SpatialData image.

    TIFF, BigTIFF/BTF, PNG, and JPEG inputs are supported. ``data_axes`` is
    the caller's explicit interpretation of the decoded array and must be a
    permutation of ``YX``, ``CYX``, ``ZYX``, or ``CZYX``. Grayscale inputs
    receive a singleton channel axis. Returned images use canonical ``CYX`` or
    ``CZYX`` dimensions without decoding pixels during construction.

    Rechunking and pyramid construction are lazy. Source decoding follows the
    TIFF storage layout, so computing a region decodes the tiles or strips of
    the output chunks it intersects; PNG and JPEG decoding is delayed but
    necessarily whole-frame.

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
    scale_factors
        Integer factors of at least two, one per pyramid level generated from
        the selected source level; empty for a single scale.
    chunks
        One positive size per canonical axis, applied at every pyramid level:
        an integer for every axis, a tuple in canonical ``CYX``/``CZYX`` order,
        or a mapping keyed by exactly the canonical axes. ``None`` selects
        ``(1, 1024, 1024)`` for ``CYX`` and ``(1, 1, 1024, 1024)`` for
        ``CZYX``: one channel and one plane per chunk with square 1024-pixel
        tiles.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        A single-scale 2D/3D SpatialData image, or a multiscale image when
        ``scale_factors`` is not empty.

    Raises
    ------
    TypeError
        If scale factors or chunk sizes are not integers, including irregular
        per-axis partitions and ``None`` sizes.
    ValueError
        If axes, TIFF selection, scale factors, chunks, or the suffix are
        invalid.

    Notes
    -----
    This function does not mutate caller-owned mappings or arrays. TIFF
    metadata inspection is eager and bounded. Raster pixel decoding is lazy.
    """
    _validate_tiff_selection(tiff_series, tiff_level)
    target_axes = _canonical_axes(data_axes)
    chunks_by_axis = _normalize_chunks(chunks, axes=target_axes)
    # Generic chunks are axis-aware and validated above; the shared options own the factors.
    model_scale_factors = make_raster_options(scale_factors).model_scale_factors

    raster = _read_raster(Path(path), tiff_series=tiff_series, tiff_level=tiff_level)
    normalized = normalize_raster_axes(
        raster.data,
        data_axes,
        target_axes=target_axes,
        add_missing=("c",) if "c" not in {axis.lower() for axis in data_axes} else (),
    )

    # Model chunking rechunks the lazy source graph without decoding pixels here.
    model = Image2DModel if target_axes == ("c", "y", "x") else Image3DModel
    return model.parse(
        normalized,
        dims=target_axes,
        transformations={coordinate_system: Identity()},
        scale_factors=model_scale_factors,
        chunks=chunks_by_axis,
    )
