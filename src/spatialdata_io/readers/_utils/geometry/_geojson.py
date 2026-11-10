"""Construct GeoJSON Polygon and MultiPolygon geometries from strict JSON without repair."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

_MIN_RING_VERTICES = 4
_SPATIAL_DIMENSIONS = 2


def _ring_violation(ring: object) -> str | None:
    """Describe the first structural violation of one decoded ring, if any."""
    # Shapely would close an open ring implicitly, so closure is checked here.
    if not isinstance(ring, list) or len(ring) < _MIN_RING_VERTICES or ring[0] != ring[-1]:
        return "an unclosed, short or malformed ring"
    for vertex in ring:
        if not isinstance(vertex, list) or len(vertex) != _SPATIAL_DIMENSIONS:
            return "a nonplanar or malformed vertex"
        # Exact types reject bool and numeric strings that NumPy would coerce.
        if any(type(value) not in (int, float) for value in vertex):
            return "a nonnumeric coordinate"
    return None


def geojson_polygons(
    geometries: Sequence[object], /, *, context: ReaderErrorContext, path: Path
) -> NDArray[np.object_]:
    """Return Shapely geometries for decoded GeoJSON geometry objects.

    Parameters
    ----------
    geometries
        Borrowed GeoJSON geometry objects decoded by the strict JSON reader
        (``read_json_object``), whose numbers are finite and within the float
        range; one per output row. They are not mutated.
    context
        Reader and artifact identity for structural errors.
    path
        Source artifact for structural errors.

    Returns
    -------
    numpy.ndarray
        One-dimensional object array of ``Polygon`` and ``MultiPolygon`` values
        in source order, with source vertices, holes and part order.

    Raises
    ------
    ReaderFormatError
        If a geometry is missing, nonpolygonal or empty, or a ring is open,
        shorter than four vertices, nonplanar or nonnumeric. Nothing is closed,
        reprojected or repaired.

    Notes
    -----
    Coordinates become float64, so integers beyond ``2**53`` may round.
    Topology is not classified; use :func:`invalid_geometry_mask` for that.
    Coordinates are validated in Python, then every geometry is built in one
    vectorized ragged-array call instead of one Shapely object per feature.
    """
    rings: list[NDArray[np.float64]] = []
    # Offsets of the ragged MultiPolygon layout: rings -> parts -> geometries.
    part_rings = [0]
    geometry_parts = [0]
    is_polygon = np.empty(len(geometries), dtype=np.bool_)
    for position, geometry in enumerate(geometries):
        if not isinstance(geometry, dict) or geometry.get("type") not in ("Polygon", "MultiPolygon"):
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"a missing or nonpolygon geometry {position}",
                expected="GeoJSON Polygon or MultiPolygon geometries",
            )
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, list) or not coordinates:
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"empty or malformed coordinates in geometry {position}",
                expected="nonempty Polygon or MultiPolygon coordinates",
            )
        is_polygon[position] = geometry["type"] == "Polygon"
        for polygon in [coordinates] if is_polygon[position] else coordinates:
            if not isinstance(polygon, list) or not polygon:
                raise ReaderFormatError(
                    context=context,
                    path=path,
                    found=f"missing polygon rings in geometry {position}",
                    expected="a shell ring followed by optional hole rings",
                )
            for ring in polygon:
                violation = _ring_violation(ring)
                if violation is not None:
                    raise ReaderFormatError(
                        context=context,
                        path=path,
                        found=f"{violation} in geometry {position}",
                        expected="closed 2D rings of at least four numeric vertices",
                    )
                rings.append(np.array(ring, dtype=np.float64))
            part_rings.append(len(rings))
        geometry_parts.append(len(part_rings) - 1)

    if not rings:
        return np.empty(0, dtype=object)
    ring_vertices = np.cumsum([0, *(len(ring) for ring in rings)], dtype=np.int64)
    result = shapely.from_ragged_array(
        shapely.GeometryType.MULTIPOLYGON,
        np.concatenate(rings),
        (ring_vertices, np.asarray(part_rings, dtype=np.int64), np.asarray(geometry_parts, dtype=np.int64)),
    )
    # Single-part collections become the source Polygon type again.
    result[is_polygon] = shapely.get_geometry(result[is_polygon], 0)
    return result
