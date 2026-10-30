"""Source polygon semantics, exact IDs and physical square footprints."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import shapely
from numpy.testing import assert_allclose, assert_array_equal

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium_hd import _geometry as geometry
from spatialdata_io.readers.visium_hd._transforms import SlideFrame

if TYPE_CHECKING:
    from pathlib import Path


def _frame() -> SlideFrame:
    return SlideFrame(np.eye(3), np.eye(3), np.eye(3), 0.0, 0.0)


def _feature(identifier: object = 1, coordinates: object = None) -> dict[str, object]:
    return {
        "type": "Feature",
        "properties": {"cell_id": identifier, "cell_centroid": [1.0, 2.0]},
        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates if coordinates is not None else [[[0, 0], [1, 0], [1, 1], [0, 0]]],
        },
    }


def _write(tmp_path: Path, features: list[dict[str, object]]) -> ArtifactFile:
    path = ArtifactFile(tmp_path / "boundaries.geojson")
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


class TestReadBoundaries:
    """Verify read boundaries contracts."""

    def test_order_vertices_ids_centroids_and_unprojected_columns(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [_feature(3), _feature(1)])
        result = geometry.read_boundaries(path, _frame(), kind="cell")
        assert_array_equal(result.shapes.index, [3, 1])
        assert result.shapes.index.dtype == np.dtype(np.int64)
        assert result.centroids is not None
        assert_allclose(result.centroids, [[3, 5], [3, 5]])
        assert_allclose(
            shapely.get_coordinates(result.shapes.geometry.to_numpy())[:4], [[1, 1], [3, 1], [3, 3], [1, 1]]
        )
        assert result.shapes.crs is None

    def test_absent_nucleus_centroid(self, tmp_path: Path) -> None:
        result = geometry.read_boundaries(_write(tmp_path, [_feature()]), _frame(), kind="nucleus")
        assert result.centroids is None

    @pytest.mark.parametrize("identifier", [None, True, 0, -1, 1.5, "1", 1_000_000_000, 2**53 + 1])
    def test_bad_ids(self, tmp_path: Path, identifier: object) -> None:
        with pytest.raises(ReaderFormatError, match="cell_id"):
            geometry.read_boundaries(_write(tmp_path, [_feature(identifier)]), _frame(), kind="cell")

    def test_duplicate_ids(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="duplicate"):
            geometry.read_boundaries(_write(tmp_path, [_feature(), _feature()]), _frame(), kind="cell")

    def test_missing_id(self, tmp_path: Path) -> None:
        feature = _feature()
        properties: dict[str, object] = {}
        feature["properties"] = properties
        with pytest.raises(ReaderFormatError, match="missing cell_id"):
            geometry.read_boundaries(_write(tmp_path, [feature]), _frame(), kind="cell")

    @pytest.mark.parametrize("mode", ["holes", "multi", "point", "null", "empty", "nonfinite", "z", "unclosed"])
    def test_malformed_geometry(self, tmp_path: Path, mode: str) -> None:
        feature = _feature()
        ring = [[0, 0], [1, 0], [1, 1], [0, 0]]
        if mode == "holes":
            feature["geometry"] = {"type": "Polygon", "coordinates": [ring, ring]}
        elif mode == "multi":
            feature["geometry"] = {"type": "MultiPolygon", "coordinates": [[ring]]}
        elif mode == "point":
            feature["geometry"] = {"type": "Point", "coordinates": [1, 2]}
        elif mode == "null":
            feature["geometry"] = None
        elif mode == "empty":
            feature["geometry"] = {"type": "Polygon", "coordinates": list[object]()}
        elif mode == "nonfinite":
            feature["geometry"] = {"type": "Polygon", "coordinates": [[[0, 0], [float("inf"), 0], [1, 1], [0, 0]]]}
        elif mode == "z":
            feature["geometry"] = {"type": "Polygon", "coordinates": [[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]]]}
        elif mode == "unclosed":
            feature["geometry"] = {"type": "Polygon", "coordinates": [ring[:-1]]}
        with pytest.raises(ReaderFormatError):
            geometry.read_boundaries(_write(tmp_path, [feature]), _frame(), kind="cell")

    def test_preserves_invalid_geometry(self, tmp_path: Path) -> None:
        ring = [[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]]
        result = geometry.read_boundaries(_write(tmp_path, [_feature(coordinates=[ring])]), _frame(), kind="cell")
        assert not result.shapes.geometry.is_valid.iloc[0]
        assert_allclose(shapely.get_coordinates(result.shapes.geometry.to_numpy()), 2 * np.array(ring) + 1)

    @pytest.mark.parametrize("value", [None, [1], [1, 2, 3], [1, float("nan")], "bad"])
    def test_bad_centroid(self, tmp_path: Path, value: object) -> None:
        feature = _feature()
        feature["properties"] = {"cell_id": 1, "cell_centroid": value}
        with pytest.raises(ReaderFormatError, match="centroid"):
            geometry.read_boundaries(_write(tmp_path, [feature]), _frame(), kind="cell")

    def test_malformed_file_and_io(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [_feature()])
        path.write_text("not geojson")
        with pytest.raises(ReaderFormatError, match="malformed"):
            geometry.read_boundaries(path, _frame(), kind="cell")
        path.unlink()
        with pytest.raises(FileNotFoundError) as error:
            geometry.read_boundaries(path, _frame(), kind="cell")
        assert error.value.__notes__


class TestBinSquares:
    """Verify bin squares contracts."""

    def test_nominal_full_edge_and_order(self) -> None:
        result = geometry.bin_squares(
            np.array([2, 0], dtype=np.int64), np.array([2, 1], dtype=np.int64), 16, _frame(), pd.Index([7, 2])
        )
        assert_array_equal(result.index, [7, 2])
        assert_allclose(result.geometry.area, [256, 256])
        assert_allclose(
            shapely.get_coordinates(result.geometry.to_numpy())[:5], [[32, 32], [48, 32], [48, 48], [32, 48], [32, 32]]
        )

    def test_empty(self) -> None:
        with pytest.raises(ReaderFormatError, match="empty bin population"):
            geometry.bin_squares(
                np.array([], dtype=np.int64), np.array([], dtype=np.int64), 2, _frame(), pd.Index([], dtype=np.int64)
            )
