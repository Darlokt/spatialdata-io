"""Preserve explicit boundary identity and physical area-derived circles."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon
from spatialdata.transformations import Identity, Scale, get_transformation

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.seqfish._geometry import make_circles, read_boundaries

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pytest_mock import MockerFixture
    from shapely.geometry.base import BaseGeometry


def _write_boundaries(
    path: Path, geometries: Sequence[BaseGeometry | None], ids: tuple[str, ...] | None
) -> ArtifactFile:
    """Write source Feature IDs independently of row positions and properties."""
    features = []
    for position, geometry in enumerate(geometries):
        identifier = f'"id":{json.dumps(ids[position])},' if ids is not None else ""
        encoded = shapely.to_geojson(geometry) if geometry is not None else "null"
        features.append(f'{{"type":"Feature",{identifier}"geometry":{encoded},"properties":{{}}}}')
    path.write_text('{"type":"FeatureCollection","features":[' + ",".join(features) + "]}", encoding="utf-8")
    return ArtifactFile(path)


class TestReadBoundaries:
    """Align source IDs exactly while preserving holes and multipart geometry."""

    def test_keep_preserves_invalid_source_topology(self, tmp_path: Path) -> None:
        invalid = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        path = _write_boundaries(tmp_path / "boundaries.geojson", [invalid], ("9",))

        result = read_boundaries(
            path, instance_ids=pd.Index([9]), roi="global", scale=Scale([0.25, 0.5], axes=("y", "x")), policy="keep"
        )

        assert result.index.tolist() == [9]
        assert shapely.equals_exact(result.geometry.iloc[0], invalid, 0)
        assert not result.geometry.is_valid.iloc[0]

    def test_already_aligned_ids_remain_in_source_order(self, tmp_path: Path) -> None:
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        path = _write_boundaries(tmp_path / "boundaries.geojson", [polygon], ("1",))
        result = read_boundaries(
            path, instance_ids=pd.Index([1]), roi="Roi1", scale=Scale([1, 1], axes=("y", "x")), policy="keep"
        )
        assert result.index.tolist() == [1]
        assert result.geometry.iloc[0].equals(polygon)

    def test_reordered_explicit_ids_preserve_geometry_and_holes(self, tmp_path: Path) -> None:
        polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]])
        multipart = MultiPolygon([Polygon([(5, 0), (6, 0), (6, 1), (5, 1)]), Polygon([(8, 0), (9, 0), (9, 1), (8, 1)])])
        path = _write_boundaries(tmp_path / "boundaries.geojson", [multipart, polygon], ("2", "70001"))
        scale = Scale([0.25, 0.5], axes=("y", "x"))

        result = read_boundaries(path, instance_ids=pd.Index([70001, 2]), roi="Roi1", scale=scale, policy="keep")

        assert result.index.tolist() == [70001, 2]
        assert result.geometry.iloc[0].equals(polygon)
        assert result.geometry.iloc[1].equals(multipart)
        assert get_transformation(result, to_coordinate_system="Roi1") == scale

    @pytest.mark.parametrize("ids", [None, ("1", "1"), ("0", "2"), ("1", "3")])
    def test_missing_duplicate_invalid_or_mismatched_ids(self, tmp_path: Path, ids: tuple[str, ...] | None) -> None:
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        path = _write_boundaries(tmp_path / "boundaries.geojson", [polygon, polygon], ids)
        with pytest.raises(ReaderFormatError):
            read_boundaries(
                path, instance_ids=pd.Index([1, 2]), roi="Roi1", scale=Scale([1, 1], axes=("y", "x")), policy="keep"
            )

    @pytest.mark.parametrize("invalid", ["null", "empty", "crossing", "line", "z", "nonfinite", "population"])
    def test_invalid_geometry_or_population_fails(self, tmp_path: Path, invalid: str) -> None:
        geometry: BaseGeometry | None = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        if invalid == "null":
            geometry = None
        elif invalid == "empty":
            geometry = Polygon()
        elif invalid == "crossing":
            geometry = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
        elif invalid == "line":
            geometry = LineString([(0, 0), (1, 1)])
        elif invalid == "z":
            geometry = Polygon([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
        elif invalid == "nonfinite":
            # GeoJSON encoding would replace infinity with null; retain the
            # malformed geometry at the storage boundary instead.
            geometry = Polygon([(0, 0), (float("inf"), 0), (1, 1), (0, 1)])
        path = _write_boundaries(tmp_path / "boundaries.geojson", [geometry], ("1",))
        with pytest.raises(ValueError, match=r"boundaries|Geometry|geometry|coordinates|polygon"):
            read_boundaries(
                path,
                instance_ids=pd.Index([1, 2] if invalid == "population" else [1]),
                roi="Roi1",
                scale=Scale([1, 1], axes=("y", "x")),
                policy="raise",
            )

    def test_empty_boundaries_fail(self, tmp_path: Path) -> None:
        path = _write_boundaries(tmp_path / "boundaries.geojson", [], ())
        with pytest.raises(ReaderFormatError, match="empty"):
            read_boundaries(
                path, instance_ids=pd.Index([1]), roi="Roi1", scale=Scale([1, 1], axes=("y", "x")), policy="keep"
            )

    def test_source_failure_keeps_error_identity_and_context(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = ArtifactFile(tmp_path / "boundaries.geojson")
        error = OSError("injected source failure")
        mocker.patch("pathlib.Path.open", side_effect=error)
        with pytest.raises(OSError, match="injected") as caught:
            read_boundaries(
                path, instance_ids=pd.Index([1]), roi="Roi1", scale=Scale([1, 1], axes=("y", "x")), policy="keep"
            )
        assert caught.value is error
        assert any("seqfish" in note and str(path) in note for note in error.__notes__)

    @pytest.mark.parametrize("property_id", ["2", None])
    def test_properties_cannot_substitute_for_top_level_identity(self, tmp_path: Path, property_id: str | None) -> None:
        path = tmp_path / "boundaries.geojson"
        identifier = '"id":"1",' if property_id is not None else ""
        path.write_text(
            '{"type":"FeatureCollection","features":[{"type":"Feature",'
            + identifier
            + '"properties":{"id":"2"},"geometry":{"type":"Polygon","coordinates":'
            + "[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}}]}",
            encoding="utf-8",
        )
        with pytest.raises(ReaderFormatError, match=r"explicit positive|ID disagreement"):
            read_boundaries(
                ArtifactFile(path),
                instance_ids=pd.Index([2]),
                roi="Roi1",
                scale=Scale([1, 1], axes=("y", "x")),
                policy="keep",
            )


class TestMakeCircles:
    """Use owned physical centroids and positive square-micrometer areas."""

    def test_exact_ids_centers_and_radii_without_table_mutation(self) -> None:
        table = ad.AnnData(obs=pd.DataFrame({"instance_id": [70001, 2], "area": [np.pi, 4 * np.pi]}))
        table.obsm["spatial"] = np.asarray([[1.5, 2.5], [4.5, 6.5]])
        original = table.copy()

        result = make_circles(table, roi="Roi1")

        assert result.index.tolist() == [70001, 2]
        np.testing.assert_array_equal(result.geometry.x, [1.5, 4.5])
        np.testing.assert_array_equal(result.geometry.y, [2.5, 6.5])
        np.testing.assert_allclose(result["radius"], [1, 2], rtol=1e-15)
        assert get_transformation(result, to_coordinate_system="Roi1") == Identity()
        pd.testing.assert_frame_equal(cast("pd.DataFrame", table.obs), cast("pd.DataFrame", original.obs))
        np.testing.assert_array_equal(table.obsm["spatial"], original.obsm["spatial"])

    @pytest.mark.parametrize("area", [0.0, -1.0])
    def test_nonpositive_circle_area_fails(self, area: float) -> None:
        table = ad.AnnData(obs=pd.DataFrame({"instance_id": [1], "area": [area]}))
        table.obsm["spatial"] = np.asarray([[1.5, 2.5]])
        with pytest.raises(ValueError, match="positive"):
            make_circles(table, roi="Roi1")
