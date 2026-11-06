"""Route one caller-selected image or GeoJSON file to its element reader by suffix.

Generic files have no dataset root, layout or format version: the suffix
selects :func:`image` or :func:`geojson`, and options of the other kind must
keep their defaults so that no argument is silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from spatialdata_io._raster_defaults import DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.geometry import validate_invalid_geometry_policy
from spatialdata_io.readers.generic._images import VALID_IMAGE_TYPES, image
from spatialdata_io.readers.generic._shapes import VALID_SHAPE_TYPES, geojson

if TYPE_CHECKING:
    from collections.abc import Sequence

    from geopandas import GeoDataFrame
    from xarray import DataArray, DataTree

    from spatialdata_io.readers.generic._images import Chunks


# Explicit public options keep format and model contracts inspectable.
def generic(  # noqa: PLR0913, RUF100
    path: str | Path,
    *,
    data_axes: Sequence[str] | None = None,
    coordinate_system: str | None = None,
    tiff_series: int | None = None,
    tiff_level: int = 0,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: Chunks | None = None,
    invalid_geometries: Literal["keep", "drop", "raise"] = "keep",
) -> DataArray | DataTree | GeoDataFrame:
    """Read one image or GeoJSON file as a SpatialData element.

    The case-insensitive suffix selects the element:

    - ``.tif``, ``.tiff``, ``.btf``, ``.png``, ``.jpg``, ``.jpeg``: a lazy
      image read by :func:`image`.
    - ``.geojson``: eager polygon shapes read by :func:`geojson`.

    No filename convention or software version is inferred.

    Parameters
    ----------
    path
        Image or GeoJSON file. ``~`` is expanded.
    data_axes
        Image only, required: the axes of the decoded array, a
        case-insensitive permutation of ``yx``, ``cyx``, ``zyx`` or ``czyx``.
    coordinate_system
        Name of the target coordinate system, preserved literally; it must
        contain a non-whitespace character. ``None`` selects ``"global"``.
        A calibrated image is in micrometres, other images in pixels and
        GeoJSON in the units of its source.
    tiff_series
        Image only: zero-based TIFF series. ``None`` reads the only series and
        fails for a TIFF with several.
    tiff_level
        Image only: zero-based stored TIFF pyramid level; zero for PNG and JPEG.
    scale_factors
        Image only: integer factors of at least two, one per generated pyramid
        level; empty for a single scale.
    chunks
        Image only: one positive chunk size per canonical axis, as an integer,
        a tuple in canonical ``CYX``/``CZYX`` order or a mapping keyed by the
        canonical axes. ``None`` selects ``(1, 1024, 1024)`` or
        ``(1, 1, 1024, 1024)``.
    invalid_geometries
        GeoJSON only: ``"keep"`` preserves invalid polygons, ``"drop"`` removes
        their rows without renumbering and ``"raise"`` fails.

    Returns
    -------
    xarray.DataArray, xarray.DataTree or geopandas.GeoDataFrame
        The element described in :func:`image` (multiscale when
        ``scale_factors`` is not empty) or :func:`geojson`, with one
        transformation to ``coordinate_system``.

    Raises
    ------
    ValueError
        If the suffix is unsupported, a required image option is missing, an
        option of the other input kind is given, or :func:`image` or
        :func:`geojson` rejects an option or the file.
    TypeError, IndexError, FileNotFoundError, IsADirectoryError
        As raised by :func:`image` or :func:`geojson`.
    """
    source = Path(path)
    suffix = source.suffix.lower()
    frame = "global" if coordinate_system is None else coordinate_system

    if suffix in VALID_SHAPE_TYPES:
        if data_axes is not None:
            message = "data_axes cannot be used with GeoJSON inputs."
            raise ValueError(message)
        if tiff_series is not None or tiff_level != 0:
            message = "tiff_series and tiff_level cannot be used with GeoJSON inputs."
            raise ValueError(message)
        if scale_factors or chunks is not None:
            message = "scale_factors and chunks cannot be used with GeoJSON inputs."
            raise ValueError(message)
        return geojson(source, coordinate_system=frame, invalid_geometries=invalid_geometries)

    if suffix in VALID_IMAGE_TYPES:
        if data_axes is None:
            message = "data_axes must be provided for image inputs."
            raise ValueError(message)
        if validate_invalid_geometry_policy(invalid_geometries) != "keep":
            message = "invalid_geometries is only applicable to GeoJSON inputs."
            raise ValueError(message)
        return image(
            source,
            data_axes=data_axes,
            coordinate_system=frame,
            tiff_series=tiff_series,
            tiff_level=tiff_level,
            scale_factors=scale_factors,
            chunks=chunks,
        )

    supported = VALID_IMAGE_TYPES + VALID_SHAPE_TYPES
    message = f"Unsupported input suffix {source.suffix!r}. Expected one of {supported}."
    raise ValueError(message)
