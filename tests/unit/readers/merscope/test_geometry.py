"""Complete projected cell footprints, independent IDs and strict geometry failures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon
from spatialdata.transformations import Identity, get_transformation

import spatialdata_io.readers.merscope._geometry as geometry_module
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.merscope._geometry import read_boundaries
from spatialdata_io.readers.merscope._schemas import STRING_DTYPE

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from pytest_mock import MockerFixture
    from shapely.geometry.base import BaseGeometry


def _write_boundaries(path: Path, geometries: list[BaseGeometry | None], ids: list[str] | list[int]) -> ArtifactFile:
    """Write WKB primitives and deliberately unprojected annotation payloads."""
    table = pa.table(
        {
            "EntityID": ids,
            "ZIndex": [0] * len(ids),
            "Type": ["Cell"] * len(ids),
            "Geometry": shapely.to_wkb(geometries),
            "unused": ["ignored"] * len(ids),
        }
    )
    pq.write_table(table, path)
    return ArtifactFile(path)


def _read(path: ArtifactFile) -> gpd.GeoDataFrame:
    return read_boundaries(path, policy="keep", frame="global").shapes


class TestReadBoundaries:
    """Union all valid components per source identity without inferring a z population."""

    def test_keep_preserves_literal_components_vertices_holes_repeats_and_order(self, tmp_path: Path) -> None:
        outer = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1, 1), (1, 2), (2, 2), (2, 1)]])
        island = Polygon([(6, 0), (7, 0), (6, 1)])
        invalid = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        other = Polygon([(10, 0), (11, 0), (10, 1)])
        path = _write_boundaries(
            tmp_path / "boundaries.parquet",
            [MultiPolygon([outer, island]), other, invalid, invalid],
            ["bad", "good", "bad", "bad"],
        )

        result = read_boundaries(path, policy="keep", frame="global")

        assert result.source_ids.tolist() == ["bad", "good"]
        assert result.shapes.index.tolist() == ["bad", "good"]
        assert result.shapes["geometry_representation"].tolist() == ["source_components", "union_footprint"]
        assert result.shapes.index.dtype == STRING_DTYPE
        geometry = result.shapes.geometry.iloc[0]
        assert isinstance(geometry, MultiPolygon)
        assert len(geometry.geoms) == 4
        for actual, expected in zip(geometry.geoms, [outer, island, invalid, invalid], strict=True):
            assert shapely.equals_exact(actual, expected, 0)
        assert not geometry.is_valid

    @pytest.mark.parametrize("policy", ["drop", "raise"])
    def test_invalid_slices_are_classified_as_whole_source_cells(
        self, tmp_path: Path, policy: Literal["drop", "raise"]
    ) -> None:
        valid = Polygon([(0, 0), (2, 0), (0, 2)])
        invalid = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        path = _write_boundaries(
            tmp_path / "boundaries.parquet", [valid, valid, invalid, invalid], ["bad", "good", "bad", "bad"]
        )
        if policy == "raise":
            with pytest.raises(ReaderFormatError, match=r"1 invalid geometries.*bad"):
                read_boundaries(path, policy=policy, frame="global")
        else:
            result = read_boundaries(path, policy=policy, frame="global")
            assert result.shapes.index.tolist() == ["good"]
            assert result.source_ids.tolist() == ["bad", "good"]

    def test_only_invalid_cells_keep_literal_components(self, tmp_path: Path) -> None:
        invalid = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        path = _write_boundaries(tmp_path / "boundaries.parquet", [invalid, invalid], ["first", "second"])
        result = read_boundaries(path, policy="keep", frame="global")
        assert result.shapes["geometry_representation"].tolist() == ["source_components", "source_components"]
        # The index representation does not depend on which cells are invalid.
        assert result.shapes.index.dtype == STRING_DTYPE
        with pytest.raises(ReaderFormatError, match="all cell boundaries dropped"):
            read_boundaries(path, policy="drop", frame="global")

    @pytest.mark.parametrize("policy", ["keep", "drop", "raise"])
    def test_empty_occupancy_slices_remain_explicitly_unsupported(
        self, tmp_path: Path, policy: Literal["keep", "drop", "raise"]
    ) -> None:
        valid = Polygon([(0, 0), (2, 0), (0, 2)])
        path = _write_boundaries(tmp_path / "boundaries.parquet", [MultiPolygon(), valid], ["cell", "cell"])
        with pytest.raises(ReaderFormatError, match="empty"):
            read_boundaries(path, policy=policy, frame="global")

    def test_invalid_union_from_valid_slices_fails_without_representation_fallback(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        valid = Polygon([(0, 0), (2, 0), (0, 2)])
        invalid = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        path = _write_boundaries(tmp_path / "boundaries.parquet", [valid], ["cell"])
        mocker.patch.object(
            geometry_module.gpd.GeoDataFrame,
            "dissolve",
            return_value=gpd.GeoDataFrame(geometry=[invalid], index=pd.Index(["cell"])),
        )
        with pytest.raises(ReaderFormatError, match="invalid projected footprints"):
            read_boundaries(path, policy="keep", frame="global")

    def test_all_slices_and_same_plane_parts_preserve_holes_multipart_and_source_order(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        outer = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]])
        island = Polygon([(6, 0), (7, 0), (7, 1), (6, 1)])
        higher = Polygon([(8, 0), (9, 0), (9, 1), (8, 1)])
        other = Polygon([(10, 0), (11, 0), (11, 1), (10, 1)])
        path = _write_boundaries(
            tmp_path / "boundaries.parquet", [outer, other, island, higher], ["001", "cell-1", "001", "001"]
        )
        table = pq.read_table(path).set_column(1, "ZIndex", pa.array([0, 2, 0, 3]))
        pq.write_table(table, path)
        read = mocker.spy(geometry_module.pq.ParquetFile, "read")

        result = _read(path)

        assert result.index.tolist() == ["001", "cell-1"]
        assert result.index.name == "EntityID"
        assert result.geometry.iloc[0].equals(MultiPolygon([outer, island, higher]))
        assert result.geometry.iloc[1].equals(other)
        assert get_transformation(result, to_coordinate_system="global") == Identity()
        assert "ZIndex" not in result
        assert "unused" not in result
        assert read.call_args.kwargs["columns"] == ["EntityID", "ZIndex", "Geometry", "Type"]

    def test_large_integer_ids_never_pass_through_float(self, tmp_path: Path) -> None:
        shape = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        path = _write_boundaries(tmp_path / "boundaries.parquet", [shape, shape], [2**53 + 1, 2**63 - 1])
        result = _read(path)
        assert result.index.tolist() == [str(2**53 + 1), str(2**63 - 1)]

    @pytest.mark.parametrize("invalid", ["missing", "empty", "3d", "line", "infinite", "wkb"])
    def test_malformed_geometry_is_never_repaired_or_dropped(self, tmp_path: Path, invalid: str) -> None:
        shape: BaseGeometry | None = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        if invalid == "missing":
            shape = None
        elif invalid == "empty":
            shape = Polygon()
        elif invalid == "3d":
            shape = Polygon([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
        elif invalid == "line":
            shape = LineString([(0, 0), (1, 1)])
        elif invalid == "infinite":
            shape = Polygon([(0, 0), (float("inf"), 0), (1, 1), (0, 1)])
        path = _write_boundaries(tmp_path / "boundaries.parquet", [shape], ["cell-1"])
        if invalid == "wkb":
            table = pq.read_table(path)
            pq.write_table(
                table.set_column(table.schema.get_field_index("Geometry"), "Geometry", pa.array([b"not WKB"])), path
            )
        with pytest.raises(ReaderFormatError, match="geometry"):
            _read(path)

    def test_parquet_failures_keep_context(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "bad.parquet")
        path.write_bytes(b"not Parquet")
        with pytest.raises(ValueError, match="Parquet") as caught:
            _read(path)
        assert any("merscope" in note and str(path) in note for note in caught.value.__notes__)

    def test_union_numerical_failure_keeps_source_context(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _write_boundaries(
            tmp_path / "boundaries.parquet", [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], ["cell-1"]
        )
        error = shapely.errors.GEOSException("injected topology failure")
        mocker.patch.object(geometry_module.gpd.GeoDataFrame, "dissolve", side_effect=error)
        with pytest.raises(ReaderFormatError, match="footprint union failed") as caught:
            _read(path)
        assert caught.value.__cause__ is error
        assert caught.value.path == path

    @pytest.mark.parametrize("fail", [False, True])
    def test_parquet_owner_closes_before_return_or_read_failure(
        self, tmp_path: Path, mocker: MockerFixture, *, fail: bool
    ) -> None:
        path = _write_boundaries(
            tmp_path / "boundaries.parquet", [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], ["cell-1"]
        )
        with pq.ParquetFile(path) as source:
            close = mocker.spy(source, "close")
            mocker.patch.object(geometry_module.pq, "ParquetFile", return_value=source)
            if fail:
                mocker.patch.object(source, "read", side_effect=RuntimeError("injected"))
                with pytest.raises(RuntimeError, match="injected"):
                    _read(path)
            else:
                result = _read(path)
                assert result.geometry.iloc[0].area == 1
            close.assert_called_once()
