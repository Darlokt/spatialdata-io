"""Unit tests for generic GeoJSON model construction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.generic._shapes import geojson
from tests.integration.readers.generic._fixtures import write_geojson

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from pytest_mock import MockerFixture
    from shapely.geometry.base import BaseGeometry


class TestGeojson:
    """Validate the shape-model boundary."""

    def test_parses_path_with_owned_transformation(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        path = tmp_path / "shapes.geojson"
        write_geojson(path, [Polygon([(0, 0), (2, 0), (0, 2)])])
        expected = object()
        parse = mocker.patch(
            "spatialdata_io.readers.generic._shapes.ShapesModel.parse",
            return_value=expected,
        )

        result = geojson(path, coordinate_system="cells")

        assert result is expected
        parse.assert_called_once()

        args, kwargs = parse.call_args
        assert args[0].index.tolist() == [0]
        assert isinstance(kwargs["transformations"]["cells"], Identity)

    @pytest.mark.parametrize("collection", [False, True])
    def test_preserves_vertices_holes_parts_and_positional_identity(self, tmp_path: Path, *, collection: bool) -> None:
        polygon = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)], holes=[[(1, 1), (1, 2), (2, 2), (2, 1)]])
        multipart = MultiPolygon([polygon, Polygon([(8, 0), (9, 0), (8, 1)])])
        source = [polygon, multipart]
        path = write_geojson(tmp_path / "shapes.geojson", source, collection=collection)

        result = geojson(path, "source")

        assert result.index.tolist() == [0, 1]
        assert result.columns.tolist() == ["geometry"]
        assert result.crs is None
        assert shapely.equals_exact(result.geometry, source, 0).all()

    @pytest.mark.parametrize("policy", ["keep", "drop", "raise"])
    def test_topology_policy(self, tmp_path: Path, policy: Literal["keep", "drop", "raise"]) -> None:
        valid = Polygon([(0, 0), (2, 0), (0, 2)])
        invalid = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        path = write_geojson(tmp_path / "shapes.geojson", [valid, invalid, valid])

        if policy == "raise":
            with pytest.raises(ReaderFormatError, match="invalid geometries"):
                geojson(path, "source", invalid_geometries=policy)
            return
        result = geojson(path, "source", invalid_geometries=policy)
        assert result.index.tolist() == ([0, 2] if policy == "drop" else [0, 1, 2])
        if policy == "keep":
            assert shapely.equals_exact(result.geometry.iloc[1], invalid, 0)

    @pytest.mark.parametrize("policy", ["keep", "drop", "raise"])
    @pytest.mark.parametrize(
        "geometry", [None, Polygon(), shapely.LineString([(0, 0), (1, 1)]), Polygon([(0, 0, 1), (2, 0, 1), (0, 2, 1)])]
    )
    def test_later_malformed_geometry_always_fails(
        self, tmp_path: Path, policy: Literal["keep", "drop", "raise"], geometry: BaseGeometry | None
    ) -> None:
        path = write_geojson(tmp_path / "shapes.geojson", [Polygon([(0, 0), (2, 0), (0, 2)]), geometry])
        with pytest.raises(ReaderFormatError):
            geojson(path, "source", invalid_geometries=policy)

    def test_all_dropped_fails(self, tmp_path: Path) -> None:
        path = write_geojson(tmp_path / "shapes.geojson", [Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])])
        with pytest.raises(ReaderFormatError, match=r"empty|all"):
            geojson(path, "source", invalid_geometries="drop")

    @pytest.mark.parametrize(
        "payload",
        [
            "{}",
            "[]",
            '{"type":"FeatureCollection","features":[]}',
            '{"type":"Polygon","coordinates":[]}',
            '{"type":"FeatureCollection","features":null}',
            '{"type":"FeatureCollection","features":[{}]}',
            '{"type":"GeometryCollection","geometries":null}',
            "{bad-json",
        ],
    )
    def test_unsupported_or_empty_document(self, tmp_path: Path, payload: str) -> None:
        path = tmp_path / "shapes.geojson"
        path.write_text(payload)
        with pytest.raises(ReaderFormatError):
            geojson(path, "source")

    def test_nonfinite_coordinates(self, tmp_path: Path) -> None:
        path = write_geojson(tmp_path / "shapes.geojson", [Polygon([(0, 0), (np.inf, 0), (0, 2)])])
        with pytest.raises(ReaderFormatError):
            geojson(path, "source")

    def test_invalid_policy_precedes_io(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalid_geometries"):
            geojson(tmp_path / "missing", "source", invalid_geometries="repair")  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.

    def test_source_failure_preserves_exception_and_context(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.geojson"
        with pytest.raises(FileNotFoundError) as caught:
            geojson(path, "source")
        assert caught.value.filename == str(path)
        assert any("generic" in note for note in caught.value.__notes__)

    def test_unsupported_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "shapes.geojson"
        path.write_bytes(b"\xff")
        with pytest.raises(ReaderFormatError, match="malformed JSON"):
            geojson(path, "source")

    # The shared GeoJSON boundary owns the complete structural permutations.
    @pytest.mark.parametrize(
        ("geometry", "found"),
        [
            ({"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [0, 2]]]}, "unclosed"),
            ({"type": "Polygon", "coordinates": [[[2**53 + 1, 0], [2, 0], [0, 2], [2**53 + 1, 0]]]}, "precision"),
        ],
    )
    def test_malformed_rings_are_never_repaired_or_rounded(
        self, tmp_path: Path, geometry: dict[str, object], found: str
    ) -> None:
        path = tmp_path / "shapes.geojson"
        path.write_text(json.dumps({"type": "GeometryCollection", "geometries": [geometry]}))
        with pytest.raises(ReaderFormatError, match=found):
            geojson(path, "source")
