"""Exact GeoJSON polygon construction and structural rejection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon, mapping

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.geometry import geojson_polygons

CONTEXT = ReaderErrorContext("synthetic", "boundaries")
PATH = Path("boundaries.geojson")
SQUARE = [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]


def _decoded(geometry: Polygon | MultiPolygon) -> object:
    """Return the standard-library decoding of a serialized geometry."""
    return json.loads(json.dumps(mapping(geometry)))


class TestGeojsonPolygons:
    def test_preserves_types_vertices_holes_part_order_and_invalid_topology(self) -> None:
        holed = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)], holes=[[(1, 1), (1, 2), (2, 2), (2, 1)]])
        multipart = MultiPolygon([Polygon([(8, 0), (9, 0), (8, 1)]), holed])
        bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        source = [holed, multipart, bowtie]
        decoded = [_decoded(geometry) for geometry in source]
        before = json.dumps(decoded)

        result = geojson_polygons(decoded, context=CONTEXT, path=PATH)

        assert result.shape == (3,)
        assert [geometry.geom_type for geometry in result] == ["Polygon", "MultiPolygon", "Polygon"]
        assert shapely.equals_exact(result, source, 0).all()
        assert not result[2].is_valid
        assert json.dumps(decoded) == before

    def test_mixed_integer_and_float_vertices_are_exact(self) -> None:
        ring = [[2**53, 0.5], [2**53 + 2, 0.5], [2**53 + 2, 4], [2**53, 0.5]]
        result = geojson_polygons([{"type": "Polygon", "coordinates": [ring]}], context=CONTEXT, path=PATH)
        assert list(result[0].exterior.coords) == [(float(x), float(y)) for x, y in ring]

    def test_empty_sequence(self) -> None:
        assert geojson_polygons([], context=CONTEXT, path=PATH).shape == (0,)

    @pytest.mark.parametrize(
        "geometry",
        [
            None,
            [],
            {"type": []},
            {"type": "Point", "coordinates": [0, 0]},
            {"type": "Polygon"},
            {"type": "Polygon", "coordinates": []},
            {"type": "Polygon", "coordinates": None},
            {"type": "MultiPolygon", "coordinates": []},
            {"type": "MultiPolygon", "coordinates": 1},
            {"type": "MultiPolygon", "coordinates": [[], [SQUARE]]},
            {"type": "MultiPolygon", "coordinates": [1]},
        ],
    )
    def test_missing_nonpolygon_or_empty_geometry(self, geometry: object) -> None:
        with pytest.raises(ReaderFormatError, match="geometry 1"):
            geojson_polygons([{"type": "Polygon", "coordinates": [SQUARE]}, geometry], context=CONTEXT, path=PATH)

    @pytest.mark.parametrize(
        ("ring", "found"),
        [
            (1, "unclosed"),
            ([[0, 0], [1, 0], [0, 0]], "short"),
            ([[0, 0], [1, 0], [1, 1], [3, 3]], "unclosed"),
            ([[0, 0], [1, 0, 0], [1, 1], [0, 0]], "nonplanar"),
            ([[0, 0], 1, [1, 1], [0, 0]], "nonplanar"),
            ([[0, 0], [True, 0], [1, 1], [0, 0]], "nonnumeric"),
            ([[False, 0], [2, 0], [0, 2], [False, 0]], "nonnumeric"),
            ([[0, 0], ["1", 0], [1, 1], [0, 0]], "nonnumeric"),
            ([[0, 0], [float("inf"), 0], [1, 1], [0, 0]], "nonfinite"),
            ([[0, 0], [float("nan"), 0], [1, 1], [0, 0]], "nonfinite"),
            ([[0, 0], [10**400, 0], [1, 1], [0, 0]], "precision"),
        ],
    )
    def test_malformed_rings_are_never_closed_or_coerced(self, ring: object, found: str) -> None:
        with pytest.raises(ReaderFormatError, match=found) as caught:
            geojson_polygons([{"type": "Polygon", "coordinates": [SQUARE, ring]}], context=CONTEXT, path=PATH)
        assert "geometry 0" in str(caught.value)
        assert "boundaries.geojson" in str(caught.value)

    @pytest.mark.parametrize("x", [2**53, 2**53 + 1, 2**53 + 2, -(2**53 + 1), 2**60])
    def test_integral_vertices_cannot_round_silently(self, x: int) -> None:
        ring = [[x, 0], [x + 4, 0], [x + 4, 4], [x, 4], [x, 0]]
        geometry = {"type": "Polygon", "coordinates": [ring]}
        if any(value != float(value) for value in (x, x + 4)):
            with pytest.raises(ReaderFormatError, match="integer precision loss"):
                geojson_polygons([geometry], context=CONTEXT, path=PATH)
        else:
            result = geojson_polygons([geometry], context=CONTEXT, path=PATH)
            assert result[0].exterior.coords[0] == (float(x), 0.0)
