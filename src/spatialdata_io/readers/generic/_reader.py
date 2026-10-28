"""Public orchestration for generic raster and GeoJSON inputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.image import DEFAULT_SCALE_FACTORS
from spatialdata_io.readers.generic._images import VALID_IMAGE_TYPES, image
from spatialdata_io.readers.generic._shapes import VALID_SHAPE_TYPES, geojson

if TYPE_CHECKING:
    from collections.abc import Sequence

    from geopandas import GeoDataFrame
    from xarray import DataArray, DataTree

    from spatialdata_io.readers.generic._images import Chunks

_DEFAULT_COORDINATE_SYSTEM = "global"


# Explicit public options keep format and model contracts inspectable.
def generic(  # noqa: PLR0913, RUF100
    path: str | Path,
    *,
    data_axes: Sequence[str] | None = None,
    coordinate_system: str | None = None,
    tiff_series: int = 0,
    tiff_level: int = 0,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: Chunks | None = None,
) -> DataArray | DataTree | GeoDataFrame:
    """Read a generic image or GeoJSON file as a SpatialData element.

    Image inputs require explicit axes and remain lazy when returned. TIFF,
    BigTIFF/BTF, PNG, and JPEG rasters are supported, including 2D and 3D
    grayscale or multichannel images. GeoJSON inputs support polygons and
    multipolygons.

    Parameters
    ----------
    path
        Input image or GeoJSON file.
    data_axes
        Explicit image axes. Required for images and invalid for GeoJSON.
    coordinate_system
        Coordinate system for the returned element. Defaults to ``"global"``.
    tiff_series
        Zero-based TIFF series. Must remain zero for non-TIFF inputs.
    tiff_level
        Zero-based TIFF pyramid level. Must remain zero for non-TIFF inputs.
    scale_factors
        Integer factors of at least two, one per generated image pyramid
        level; empty for a single scale. Invalid for GeoJSON inputs.
    chunks
        One positive image chunk size per canonical axis, applied at every
        pyramid level: an integer for every axis, a tuple in canonical
        ``CYX``/``CZYX`` order, or a mapping keyed by exactly the canonical
        axes. ``None`` selects ``(1, 1024, 1024)`` for ``CYX`` and
        ``(1, 1, 1024, 1024)`` for ``CZYX``: one channel and one plane per
        chunk with square 1024-pixel tiles. Invalid for GeoJSON inputs.

    Returns
    -------
    xarray.DataArray, xarray.DataTree, or geopandas.GeoDataFrame
        Parsed SpatialData element with an identity transformation. Images are
        multiscale ``DataTree`` values only when ``scale_factors`` is not empty.

    Raises
    ------
    TypeError
        If image scale factors or chunk sizes are not integers.
    ValueError
        If the suffix or format-specific arguments are invalid.
    """
    normalized_path = Path(path)
    suffix = normalized_path.suffix.lower()
    target_coordinate_system = coordinate_system or _DEFAULT_COORDINATE_SYSTEM

    if suffix in VALID_SHAPE_TYPES:
        if data_axes is not None:
            message = "data_axes cannot be used with GeoJSON inputs."
            raise ValueError(message)
        if tiff_series != 0 or tiff_level != 0:
            message = "tiff_series and tiff_level cannot be used with GeoJSON inputs."
            raise ValueError(message)
        if scale_factors or chunks is not None:
            message = "scale_factors and chunks cannot be used with GeoJSON inputs."
            raise ValueError(message)
        return geojson(normalized_path, coordinate_system=target_coordinate_system)

    if suffix in VALID_IMAGE_TYPES:
        if data_axes is None:
            message = "data_axes must be provided for image inputs."
            raise ValueError(message)
        return image(
            normalized_path,
            data_axes=data_axes,
            coordinate_system=target_coordinate_system,
            tiff_series=tiff_series,
            tiff_level=tiff_level,
            scale_factors=scale_factors,
            chunks=chunks,
        )

    supported = VALID_IMAGE_TYPES + VALID_SHAPE_TYPES
    message = f"Unsupported input suffix {normalized_path.suffix!r}. Expected one of {supported}."
    raise ValueError(message)
