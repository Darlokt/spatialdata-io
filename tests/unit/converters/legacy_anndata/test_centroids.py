from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from dask.callbacks import Callback
from geopandas import GeoDataFrame, GeoSeries
from shapely import Polygon
from spatialdata import SpatialData
from spatialdata.models import Labels2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Affine, Identity, Scale, get_transformation, set_transformation

from spatialdata_io.converters.legacy_anndata._centroids import centroid_coordinates, linked_targets

if TYPE_CHECKING:
    from pathlib import Path


def _table() -> AnnData:
    table = AnnData(
        shape=(2, 0),
        obs=pd.DataFrame(
            {"region": pd.Categorical(["cells", "cells"]), "instance": [11, 5]},
            index=["cell_eleven", "cell_five"],
        ),
    )
    return TableModel.parse(table, region="cells", region_key="region", instance_key="instance")


def _scene(kind: Literal["shapes", "points", "labels"] = "shapes") -> tuple[SpatialData, AnnData]:
    table = _table()
    transform = Affine(
        np.array([[2.0, 1.0, 10.0], [0.0, -3.0, 20.0], [0.0, 0.0, 1.0]]),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    if kind == "shapes":
        shapes = ShapesModel.parse(
            GeoDataFrame(
                geometry=[Polygon([(0, 0), (6, 0), (0, 3)]), Polygon([(6, 3), (12, 3), (6, 6)])],
                index=pd.Index([5, 11]),
            ),
            transformations={"analysis": transform},
        )
        return SpatialData(shapes={"cells": shapes}, tables={"table": table}), table
    if kind == "points":
        points = PointsModel.parse(
            pd.DataFrame({"x": [2, 8], "y": [1, 4], "annotation": ["first", "second"]}, index=pd.Index([5, 11])),
            transformations={"analysis": transform},
        )
        return SpatialData(points={"cells": points}, tables={"table": table}), table
    pixels = np.zeros((4, 5), dtype=np.uint16)
    pixels[0, 3] = 11
    pixels[2:4, 0:2] = 5
    labels = Labels2DModel.parse(pixels, dims=("y", "x"), chunks=(2, 3), transformations={"analysis": transform})
    return SpatialData(labels={"cells": labels}, tables={"table": table}), table


class TestLinkedTargets:
    @pytest.mark.parametrize("kind", ["shapes", "points", "labels"])
    def test_preserves_table_order_without_computing_lazy_elements(
        self, kind: Literal["shapes", "points", "labels"]
    ) -> None:
        sdata, table = _scene(kind)
        tasks: list[str] = []
        with Callback(pretask=lambda key, _graph, _state: tasks.append(str(key))):
            targets = linked_targets(sdata, table)
        assert tasks == []
        assert len(targets) == 1
        assert targets[0].name == "cells"
        assert targets[0].instances.tolist() == [11, 5]
        np.testing.assert_array_equal(targets[0].rows, [0, 1])
        assert targets[0].element is sdata["cells"]

    def test_accepts_multiple_regions_with_same_instance_ids(self) -> None:
        sdata, table = _scene("points")
        sdata.points["other"] = sdata.points["cells"].copy()
        table = AnnData(
            shape=(4, 0),
            obs=pd.DataFrame(
                {"region": pd.Categorical(["other", "cells", "other", "cells"]), "instance": [11, 5, 5, 11]},
                index=["a", "b", "c", "d"],
            ),
        )
        table = TableModel.parse(table, region=["cells", "other"], region_key="region", instance_key="instance")
        targets = linked_targets(sdata, table)
        assert tuple(target.name for target in targets) == ("other", "cells")
        np.testing.assert_array_equal(targets[0].rows, [0, 2])
        np.testing.assert_array_equal(targets[1].rows, [1, 3])

    def test_accepts_persisted_multiregion_metadata(self, tmp_path: Path) -> None:
        sdata, table = _scene("points")
        sdata.points["other"] = sdata.points["cells"].copy()
        table.obs["region"] = pd.Categorical(["cells", "other"])
        table.uns[TableModel.ATTRS_KEY]["region"] = ["cells", "other"]
        path = tmp_path / "linked.h5ad"
        table.write_h5ad(path)
        restored = read_h5ad(path)
        assert isinstance(restored.uns[TableModel.ATTRS_KEY]["region"], np.ndarray)
        targets = linked_targets(sdata, restored)
        assert tuple(target.name for target in targets) == ("cells", "other")

    @pytest.mark.parametrize("field", ["region_key", "instance_key"])
    @pytest.mark.parametrize("value", [None, "", 1, "missing"])
    def test_rejects_invalid_linkage_column_declarations(self, field: str, value: object) -> None:
        sdata, table = _scene()
        table.uns[TableModel.ATTRS_KEY][field] = value
        with pytest.raises(ValueError, match=r"region|instance|column|spatialdata_attrs"):
            linked_targets(sdata, table)

    def test_rejects_same_region_and_instance_column(self) -> None:
        sdata, table = _scene()
        table.uns[TableModel.ATTRS_KEY]["instance_key"] = "region"
        with pytest.raises(ValueError, match="distinct"):
            linked_targets(sdata, table)

    @pytest.mark.parametrize("regions", [1, [], [""], [1], np.array([["cells"]])])
    def test_rejects_invalid_declared_regions(self, regions: object) -> None:
        sdata, table = _scene()
        table.uns[TableModel.ATTRS_KEY]["region"] = regions
        with pytest.raises(ValueError, match=r"target|region|spatialdata_attrs"):
            linked_targets(sdata, table)

    @pytest.mark.parametrize("metadata", [None, [], {"region": "cells"}])
    def test_rejects_malformed_linkage_metadata(self, metadata: object) -> None:
        sdata, table = _scene()
        table.uns[TableModel.ATTRS_KEY] = metadata
        with pytest.raises(ValueError, match=r"spatialdata_attrs|linkage|table keys"):
            linked_targets(sdata, table)

    @pytest.mark.parametrize("field", ["region", "instance"])
    def test_rejects_null_observed_identifiers(self, field: str) -> None:
        sdata, table = _scene()
        table.obs[field] = [None, 5] if field == "instance" else [None, "cells"]
        with pytest.raises(ValueError, match="non-null"):
            linked_targets(sdata, table)

    @pytest.mark.parametrize("ids", [[1.0, 2.0], [True, False], [5, "eleven"]])
    def test_rejects_ambiguous_instance_dtypes(self, ids: list[int | str | float | bool]) -> None:
        sdata, table = _scene()
        table.obs["instance"] = ids
        with pytest.raises(ValueError, match="integer or string"):
            linked_targets(sdata, table)

    def test_rejects_undeclared_observed_region(self) -> None:
        sdata, table = _scene()
        table.uns[TableModel.ATTRS_KEY]["region"] = ["other"]
        with pytest.raises(ValueError, match="declared non-null"):
            linked_targets(sdata, table)

    def test_rejects_duplicate_region_instance_pairs(self) -> None:
        sdata, table = _scene()
        table.obs["instance"] = [5, 5]
        with pytest.raises(ValueError, match=r"unique.*region.*instance"):
            linked_targets(sdata, table)

    def test_rejects_absent_target(self) -> None:
        sdata, table = _scene()
        del sdata.shapes["cells"]
        with pytest.raises(ValueError, match="available Shapes, Labels or Points"):
            linked_targets(sdata, table)

    @pytest.mark.parametrize(
        "index", [pd.Index([5, 5]), pd.Index([5, None]), pd.MultiIndex.from_tuples([(5, "a"), (11, "b")])]
    )
    def test_rejects_ambiguous_shape_identifiers(self, index: pd.Index) -> None:
        sdata, table = _scene()
        sdata.shapes["cells"].index = index
        with pytest.raises(ValueError, match="unique non-null"):
            linked_targets(sdata, table)

    def test_rejects_missing_referenced_shape(self) -> None:
        sdata, table = _scene()
        table.obs["instance"] = [99, 5]
        with pytest.raises(ValueError, match="missing referenced"):
            linked_targets(sdata, table)

    def test_accepts_exact_string_identifiers(self) -> None:
        sdata, table = _scene()
        sdata.shapes["cells"].index = pd.Index(["0005", "0011"])
        table.obs["instance"] = ["0011", "0005"]
        targets = linked_targets(sdata, table)
        assert targets[0].instances.tolist() == ["0011", "0005"]


class TestCentroidCoordinates:
    @pytest.mark.parametrize("kind", ["shapes", "points", "labels"])
    def test_applies_affine_transform_and_restores_table_order(
        self, kind: Literal["shapes", "points", "labels"]
    ) -> None:
        sdata, table = _scene(kind)
        targets = linked_targets(sdata, table)
        original = get_transformation(targets[0].element, "analysis")
        coordinates = centroid_coordinates(targets, frame="analysis", n_obs=2)
        expected = [[17.5, 18.5], [15.0, 11.0]] if kind == "labels" else [[30.0, 8.0], [15.0, 17.0]]
        np.testing.assert_array_equal(coordinates, expected)
        assert coordinates.dtype == np.float64
        assert get_transformation(targets[0].element, "analysis") == original
        assert table.obs_names.tolist() == ["cell_eleven", "cell_five"]

    def test_empty_population(self) -> None:
        coordinates = centroid_coordinates((), frame="analysis", n_obs=0)
        assert coordinates.shape == (0, 2)
        assert coordinates.dtype == np.float64

    def test_missing_frame_fails_before_label_tasks(self) -> None:
        sdata, table = _scene("labels")
        tasks: list[str] = []
        with (
            Callback(pretask=lambda key, _graph, _state: tasks.append(str(key))),
            pytest.raises(ValueError, match="no transformation"),
        ):
            centroid_coordinates(linked_targets(sdata, table), frame="missing", n_obs=2)
        assert tasks == []

    def test_rejects_three_dimensional_targets_before_computing(self) -> None:
        sdata, table = _scene("points")
        points = PointsModel.parse(
            pd.DataFrame({"x": [2.0, 8.0], "y": [1.0, 4.0], "z": [5.0, 6.0]}, index=pd.Index([5, 11])),
            transformations={"analysis": Identity()},
        )
        sdata.points["cells"] = points
        tasks: list[str] = []
        with (
            Callback(pretask=lambda key, _graph, _state: tasks.append(str(key))),
            pytest.raises(ValueError, match="XY axes"),
        ):
            centroid_coordinates(linked_targets(sdata, table), frame="analysis", n_obs=2)
        assert tasks == []

    @pytest.mark.parametrize("kind", ["points", "labels"])
    def test_rejects_missing_lazy_target_identifiers(self, kind: Literal["points", "labels"]) -> None:
        sdata, table = _scene(kind)
        table.obs["instance"] = [99, 5]
        targets = linked_targets(sdata, table)
        with pytest.raises(ValueError, match="missing referenced"):
            centroid_coordinates(targets, frame="analysis", n_obs=2)

    @pytest.mark.parametrize("index", [pd.Index([5, 5]), pd.Index([5, None])])
    def test_rejects_ambiguous_point_identifiers(self, index: pd.Index) -> None:
        sdata, table = _scene("points")
        sdata.points["cells"] = PointsModel.parse(
            pd.DataFrame({"x": [2, 8], "y": [1, 4]}, index=index), transformations={"analysis": Identity()}
        )
        with pytest.raises(ValueError, match="unique non-null"):
            centroid_coordinates(linked_targets(sdata, table), frame="analysis", n_obs=2)

    @pytest.mark.parametrize("transform", [Identity(), Scale([2.0, 1.0], axes=("x", "y"))])
    def test_rejects_lossy_large_integer_points_before_transform(self, transform: Identity | Scale) -> None:
        sdata, table = _scene("points")
        sdata.points["cells"] = PointsModel.parse(
            pd.DataFrame({"x": [2**53 + 1, 2], "y": [1, 4]}, index=pd.Index([5, 11])),
            transformations={"analysis": transform},
        )
        with pytest.raises(ValueError, match="losslessly"):
            centroid_coordinates(linked_targets(sdata, table), frame="analysis", n_obs=2)

    def test_accepts_exactly_representable_large_integer_points(self) -> None:
        sdata, table = _scene("points")
        sdata.points["cells"] = PointsModel.parse(
            pd.DataFrame({"x": [2**53 + 2, 2], "y": [1, 4]}, index=pd.Index([5, 11])),
            transformations={"analysis": Identity()},
        )
        actual = centroid_coordinates(linked_targets(sdata, table), frame="analysis", n_obs=2)
        np.testing.assert_array_equal(actual, [[2.0, 4.0], [float(2**53 + 2), 1.0]])

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite_source_points(self, value: float) -> None:
        sdata, table = _scene("points")
        sdata.points["cells"] = PointsModel.parse(
            pd.DataFrame({"x": [value, 2.0], "y": [1.0, 4.0]}, index=pd.Index([5, 11])),
            transformations={"analysis": Identity()},
        )
        with pytest.raises(ValueError, match="finite real numerical"):
            centroid_coordinates(linked_targets(sdata, table), frame="analysis", n_obs=2)

    def test_rejects_empty_referenced_polygon_with_target_context(self) -> None:
        sdata, table = _scene()
        shapes = sdata.shapes["cells"]
        shapes["geometry"] = GeoSeries([Polygon(), shapes.geometry.iloc[1]], index=shapes.index)
        with pytest.raises(ValueError, match=r"Shape|Length|length") as error:
            centroid_coordinates(linked_targets(sdata, table), frame="analysis", n_obs=2)
        assert any("cells" in note and "analysis" in note for note in error.value.__notes__)

    @pytest.mark.parametrize("scale", [np.nan, np.inf, 0.0])
    def test_rejects_nonfinite_or_singular_transform_before_label_tasks(self, scale: float) -> None:
        sdata, table = _scene("labels")
        set_transformation(sdata.labels["cells"], Scale([scale, 1.0], axes=("x", "y")), to_coordinate_system="analysis")
        tasks: list[str] = []
        with (
            Callback(pretask=lambda key, _graph, _state: tasks.append(str(key))),
            pytest.raises(ValueError, match=r"finite|invertible|singular"),
        ):
            centroid_coordinates(linked_targets(sdata, table), frame="analysis", n_obs=2)
        assert tasks == []

    def test_rejects_nonfinite_transformed_coordinates(self) -> None:
        sdata, table = _scene("points")
        sdata.points["cells"] = PointsModel.parse(
            pd.DataFrame({"x": [1e308, 2.0], "y": [1.0, 4.0]}, index=pd.Index([5, 11])),
            transformations={"analysis": Scale([2.0, 1.0], axes=("x", "y"))},
        )
        with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError, match="finite"):
            centroid_coordinates(linked_targets(sdata, table), frame="analysis", n_obs=2)
