"""Exercise explicit coordinate selection and owned legacy imports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad, settings
from anndata import read_zarr as read_anndata_zarr
from anndata.tests.helpers import assert_equal
from dask.callbacks import Callback
from scipy.sparse import csr_array
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Labels2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Affine, BaseTransformation, get_transformation

from spatialdata_io import codex, curio
from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.converters.legacy_anndata import from_legacy_anndata, to_legacy_anndata
from tests.integration.readers.codex._fixtures import write_csv
from tests.integration.readers.curio._fixtures import write_curio

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _table() -> AnnData:
    counts = csr_array([[1, 0], [0, 2], [3, 4]], dtype=np.int16)
    table = AnnData(
        counts,
        obs=pd.DataFrame(
            {
                "annotation": pd.Categorical(["a", "b", "a"]),
                "region": pd.Categorical(["cells"] * 3),
                "instance_id": [9, 3, 5],
            },
            index=["cell9", "cell3", "cell5"],
        ),
        var=pd.DataFrame({"feature_kind": ["gene", "control"]}, index=["GeneB", "GeneA"]),
    )
    table.layers["counts"] = counts.copy()
    table.layers["dense_annotation"] = np.arange(6, dtype=np.float32).reshape(3, 2)
    table.raw = table.copy()
    table.obsm["spatial"] = np.array([[100.0, 101.0], [200.0, 201.0], [300.0, 301.0]])
    table.obsm["spatial_pixels"] = np.array([[10, 11], [20, 21], [30, 31]], dtype=np.int64)
    table.uns["experiment"] = {"name": "synthetic", "settings": [1, 2]}
    _declare(table, "spatial", "micrometer", "global")
    return table


def _declare(table: AnnData, key: str, units: str, frame: str) -> None:
    description: dict[str, object] = {
        "axes": ["x", "y"],
        "units": units,
        "frame": frame,
        "position": "source_cell_center",
    }
    if units == "micrometer":
        description["calibration"] = {frame: "synthetic: source coordinates calibrated in micrometres"}
    table.uns[POSITIONS_KEY] = {"schema_version": 1, "coordinates": {key: description}}


def _linked(kind: Literal["shapes", "labels", "points"] = "shapes") -> SpatialData:
    table = _table()
    table = TableModel.parse(table, region="cells", region_key="region", instance_key="instance_id")
    transform = Affine(
        np.array([[2.0, 0.0, 10.0], [0.0, -3.0, 20.0], [0.0, 0.0, 1.0]]), input_axes=("x", "y"), output_axes=("x", "y")
    )
    if kind == "shapes":
        shapes = ShapesModel.parse(
            np.array([[2.0, 1.0], [8.0, 4.0], [5.0, 3.0]]),
            geometry=0,
            radius=2.0,
            index=pd.Index([3, 9, 5]),
            transformations={"global": transform},
        )
        return SpatialData(shapes={"cells": shapes}, tables={"table": table})
    if kind == "points":
        points = PointsModel.parse(
            pd.DataFrame({"x": [2.0, 8.0, 5.0], "y": [1.0, 4.0, 3.0]}, index=pd.Index([3, 9, 5])),
            transformations={"global": transform},
        )
        return SpatialData(points={"cells": points}, tables={"table": table})
    pixels = np.zeros((6, 10), dtype=np.uint16)
    pixels[1:3, 2:4] = 3
    pixels[3:5, 7:10] = 9
    pixels[5, 5] = 5
    labels = Labels2DModel.parse(pixels, dims=("y", "x"), chunks=(3, 5), transformations={"global": transform})
    return SpatialData(labels={"cells": labels}, tables={"table": table})


def _assert_expression(actual: AnnData, expected: AnnData) -> None:
    assert isinstance(actual.X, csr_array)
    assert actual.X.dtype == np.int16
    assert actual.obs_names.tolist() == expected.obs_names.tolist()
    assert actual.var_names.tolist() == expected.var_names.tolist()
    assert_equal(actual.X, expected.X)
    assert_equal(actual.layers, expected.layers)
    assert_equal(actual.raw, expected.raw)


class TestToLegacyAnndata:
    def test_default_copies_a_table_without_consuming_absent_targets(self) -> None:
        table = _linked().tables["table"]
        with pytest.warns(UserWarning, match="not present"):
            source = SpatialData(tables={"table": table})

        actual = to_legacy_anndata(source)

        assert_equal(actual, table)
        assert actual is not table

    def test_default_preserves_physical_table_with_no_spatial_tasks(self) -> None:
        source = _linked("labels")
        original = source.tables["table"].copy()
        tasks: list[object] = []
        with Callback(posttask=lambda key, *_args: tasks.append(key)):
            actual = to_legacy_anndata(source)

        assert not tasks
        assert_equal(actual, original)
        assert actual is not source.tables["table"]
        assert isinstance(actual.X, csr_array)
        actual.X.data[0] = 90
        actual.obsm["spatial"][0, 0] = 90
        actual.uns["experiment"]["settings"][0] = 90
        assert_equal(source.tables["table"], original)

    @pytest.mark.parametrize("kind", ["shapes", "labels", "points"])
    def test_centroids_apply_complete_transform_and_align_original_ids(
        self, kind: Literal["shapes", "labels", "points"]
    ) -> None:
        source = _linked(kind)
        original = source.tables["table"].copy()

        actual = to_legacy_anndata(source, position_source="centroids", coordinate_system="global")

        expected = (
            [[27.0, 8.0], [16.0, 14.0], [21.0, 3.5]] if kind == "labels" else [[26.0, 8.0], [14.0, 17.0], [20.0, 11.0]]
        )
        np.testing.assert_array_equal(actual.obsm["spatial"], expected)
        np.testing.assert_array_equal(actual.obsm["spatial_original"], original.obsm["spatial"])
        np.testing.assert_array_equal(actual.obsm["spatial_pixels"], original.obsm["spatial_pixels"])
        assert (
            actual.uns[POSITIONS_KEY]["coordinates"]["spatial_original"]
            == original.uns[POSITIONS_KEY]["coordinates"]["spatial"]
        )
        _assert_expression(actual, original)
        assert_equal(source.tables["table"], original)

    @pytest.mark.parametrize(
        ("units", "destination"), [("pixel", "spatial_pixels"), ("grid", "spatial_grid"), ("unknown", "spatial_source")]
    )
    def test_nonphysical_canonical_positions_are_preserved_under_semantic_key(
        self, units: str, destination: str
    ) -> None:
        table = _table()
        del table.obsm["spatial_pixels"]
        xy = np.array([[2**53 + 1, 2], [3, 4], [5, 6]], dtype=np.int64)
        table.obsm["spatial"] = xy
        _declare(table, "spatial", units, "source")

        actual = to_legacy_anndata(SpatialData(tables={"table": table}))

        assert "spatial" not in actual.obsm
        np.testing.assert_array_equal(actual.obsm[destination], xy)
        assert actual.obsm[destination].dtype == np.int64
        assert actual.uns[POSITIONS_KEY]["coordinates"][destination]["units"] == units
        assert actual.uns[POSITIONS_KEY]["coordinates"][destination]["frame"] == "source"

    def test_undeclared_coordinates_remain_unknown_and_native_only_tables_remain_native(self) -> None:
        table = _table()
        del table.uns[POSITIONS_KEY]
        actual = to_legacy_anndata(SpatialData(tables={"table": table}))
        assert "spatial" not in actual.obsm
        np.testing.assert_array_equal(actual.obsm["spatial_source"], table.obsm["spatial"])
        del table.obsm["spatial"]
        native = to_legacy_anndata(SpatialData(tables={"table": table}))
        assert "spatial" not in native.obsm
        np.testing.assert_array_equal(native.obsm["spatial_pixels"], table.obsm["spatial_pixels"])

    def test_native_centroids_do_not_replace_source_positions(self) -> None:
        source = _linked()
        table = source.tables["table"]
        del table.obsm["spatial"]
        _declare(table, "spatial_pixels", "pixel", "global")

        actual = to_legacy_anndata(source, position_source="centroids", coordinate_system="global")

        assert "spatial" not in actual.obsm
        np.testing.assert_array_equal(actual.obsm["spatial_centroids"], [[26, 8], [14, 17], [20, 11]])
        np.testing.assert_array_equal(actual.obsm["spatial_pixels"], table.obsm["spatial_pixels"])

    def test_centroids_combine_shapes_and_points_by_region_and_instance(self) -> None:
        source = _linked()
        shapes = source.shapes["cells"].loc[[9]].copy()
        transform = get_transformation(shapes, "global")
        assert isinstance(transform, BaseTransformation)
        points = PointsModel.parse(
            pd.DataFrame({"x": [5.0, 2.0], "y": [3.0, 1.0]}, index=pd.Index([5, 3])),
            transformations={"global": transform},
        )
        table = _table()
        table.obs["region"] = pd.Categorical(["cells", "detections", "detections"])
        table = TableModel.parse(table, region=["cells", "detections"], region_key="region", instance_key="instance_id")
        source = SpatialData(shapes={"cells": shapes}, points={"detections": points}, tables={"table": table})

        actual = to_legacy_anndata(source, position_source="centroids", coordinate_system="global")

        np.testing.assert_array_equal(actual.obsm["spatial"], [[26, 8], [14, 17], [20, 11]])
        assert actual.obs_names.tolist() == ["cell9", "cell3", "cell5"]
        _assert_expression(actual, table)

    def test_independent_physical_frames_require_explicit_population_selection(self) -> None:
        table = _table()
        table.obs["spatial_frame"] = pd.Categorical(["a", "b", "a"])
        declaration = table.uns[POSITIONS_KEY]["coordinates"]["spatial"]
        declaration.pop("frame")
        declaration["frame_key"] = "spatial_frame"
        declaration["calibration"] = {"a": "independent field a calibration", "b": "independent field b calibration"}
        source = SpatialData(tables={"table": table})
        with pytest.raises(ValueError, match=r"frame|coordinate_system"):
            to_legacy_anndata(source)

        actual = to_legacy_anndata(source, coordinate_system="a")

        assert actual.obs_names.tolist() == ["cell9", "cell5"]
        np.testing.assert_array_equal(actual.obsm["spatial"], table.obsm["spatial"][[0, 2]])
        _assert_expression(actual, table[["cell9", "cell5"]].copy())

    @pytest.mark.parametrize("collision", ["spatial_original", "spatial_centroids"])
    def test_centroid_destination_collisions_fail(self, collision: str) -> None:
        source = _linked()
        table = source.tables["table"]
        if collision == "spatial_centroids":
            del table.obsm["spatial"]
            _declare(table, "spatial_pixels", "pixel", "global")
        table.obsm[collision] = np.ones((3, 2))
        with pytest.raises(ValueError, match=collision):
            to_legacy_anndata(source, position_source="centroids", coordinate_system="global")

    def test_missing_referenced_target_id_fails_without_dropping_rows(self) -> None:
        source = _linked()
        observations = source.tables["table"].obs
        assert isinstance(observations, pd.DataFrame)
        observations.loc["cell9", "instance_id"] = 99
        original = source.tables["table"].copy()
        with pytest.raises(ValueError, match=r"99|missing|target"):
            to_legacy_anndata(source, position_source="centroids", coordinate_system="global")
        assert_equal(source.tables["table"], original)


class TestFromLegacyAnndata:
    def test_complete_conversions_keep_large_expression_sparse(self, mocker: MockerFixture) -> None:
        observations, features = 4000, 10_000
        values = np.arange(1, observations * 2 + 1, dtype=np.int32)
        indices = np.tile([0, features - 1], observations)
        indptr = np.arange(0, observations * 2 + 1, 2)
        counts = csr_array((values, indices, indptr), shape=(observations, features))
        source = AnnData(counts)
        source.obsm["spatial"] = np.column_stack((np.arange(observations), np.arange(observations) * 2))
        mocker.patch.object(csr_array, "toarray", side_effect=AssertionError("Expression must remain sparse"))
        mocker.patch.object(csr_array, "todense", side_effect=AssertionError("Expression must remain sparse"))

        imported = from_legacy_anndata(source)
        exported = to_legacy_anndata(imported)

        for table in (imported.tables["table"], exported):
            assert isinstance(table.X, csr_array)
            assert table.X.shape == (observations, features)
            np.testing.assert_array_equal(table.X.data, values)
            np.testing.assert_array_equal(table.X.indices, indices)
            np.testing.assert_array_equal(table.X.indptr, indptr)

    def test_plain_legacy_pixels_preserve_vendor_fields_and_generate_points(self) -> None:
        source = _table()
        del source.obsm["spatial_pixels"]
        del source.uns[POSITIONS_KEY]
        original = source.copy()

        result = from_legacy_anndata(source)

        assert list(result.points) == ["locations"]
        assert not result.shapes
        table = result.tables["table"]
        assert "spatial" not in table.obsm
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], original.obsm["spatial"])
        assert table.obs["region"].tolist() == original.obs["region"].tolist()
        assert table.obs["instance_id"].tolist() == [9, 3, 5]
        assert table.obs["spatialdata_region"].tolist() == ["locations"] * 3
        assert table.obs["spatialdata_instance_id"].tolist() == [0, 1, 2]
        points = result.points["locations"].compute()
        np.testing.assert_array_equal(points[["x", "y"]].to_numpy(), original.obsm["spatial"])
        _assert_expression(table, original)
        assert isinstance(table.X, csr_array)
        table.X.data[0] = 90
        table.obsm["spatial_pixels"][0, 0] = 90
        assert_equal(source, original)

    def test_declared_physical_positions_without_images_keep_their_units(self) -> None:
        source = _table()
        original = source.copy()
        result = from_legacy_anndata(source)
        table = result.tables["table"]
        np.testing.assert_array_equal(table.obsm["spatial"], original.obsm["spatial"])
        assert table.uns[POSITIONS_KEY]["coordinates"]["spatial"]["units"] == "micrometer"
        np.testing.assert_array_equal(result.points["locations"].compute()[["x", "y"]], original.obsm["spatial"])
        _assert_expression(table, original)
        assert_equal(source, original)

    def test_selected_library_subsets_original_rows_and_imports_only_its_images(self) -> None:
        source = _table()
        del source.obsm["spatial_pixels"]
        del source.uns[POSITIONS_KEY]
        source.obs["library"] = pd.Categorical(["b", "a", "b"])
        image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        source.uns["spatial"] = {
            name: {"images": {"hires": image + i}, "scalefactors": {"tissue_hires_scalef": 0.5}}
            for i, name in enumerate(("a", "b"))
        }
        original = source.copy()

        result = from_legacy_anndata(source, library_id="b", library_key="library")

        table = result.tables["table"]
        assert table.obs_names.tolist() == ["cell9", "cell5"]
        assert table.obs["spatialdata_instance_id"].tolist() == [0, 2]
        assert set(result.images) == {"b_hires"}
        imported = result.images["b_hires"]
        np.testing.assert_array_equal(imported.data.compute(), (image + 1).transpose(2, 0, 1))
        frames = get_transformation(imported, get_all=True)
        assert isinstance(frames, dict)
        point_frames = get_transformation(result.points["locations"], get_all=True)
        assert isinstance(point_frames, dict)
        shared = set(frames) & set(point_frames)
        assert len(shared) == 1
        transform = frames[shared.pop()]
        assert isinstance(transform, BaseTransformation)
        affine = transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        np.testing.assert_array_equal(affine, [[2, 0, -1], [0, 2, -1], [0, 0, 1]])
        # Intrinsic centre (.5,.5) maps to legacy array index (0,0).
        np.testing.assert_array_equal(affine @ [0.5, 0.5, 1], [0, 0, 1])
        _assert_expression(table, original[["cell9", "cell5"]].copy())
        assert_equal(source, original)

    def test_no_positions_retains_an_owned_unlinked_table_and_none_expression(self) -> None:
        source = AnnData(obs=pd.DataFrame(index=["a", "b"]))
        source.obs["vendor_id"] = [17, 29]
        source.uns[TableModel.ATTRS_KEY] = {"region": ["old"], "region_key": "region", "instance_key": "id"}
        result = from_legacy_anndata(source)
        table = result.tables["table"]
        assert table.X is None
        assert not result.points
        assert not result.shapes
        assert TableModel.ATTRS_KEY not in table.uns
        assert table.obs["vendor_id"].tolist() == [17, 29]
        assert TableModel.ATTRS_KEY in source.uns

    @pytest.mark.parametrize("storage", ["h5ad", "anndata_zarr", "spatialdata_zarr"])
    def test_actual_import_persists_coordinates_linkage_and_payload(self, tmp_path: Path, storage: str) -> None:
        source = _table()
        result = from_legacy_anndata(source)
        table = result.tables["table"]
        path = tmp_path / storage
        if storage == "h5ad":
            table.write_h5ad(path)
            with settings.override(use_sparse_array_on_read=True):
                actual = read_h5ad(path)
        elif storage == "anndata_zarr":
            table.write_zarr(path)
            with settings.override(use_sparse_array_on_read=True):
                actual = read_anndata_zarr(path)
        else:
            result.write(path)
            with settings.override(use_sparse_array_on_read=True):
                restored = read_zarr(path)
            assert list(restored.points) == ["locations"]
            np.testing.assert_array_equal(
                restored.points["locations"].compute()[["x", "y"]], result.points["locations"].compute()[["x", "y"]]
            )
            actual = restored.tables["table"]
        _assert_expression(actual, table)
        np.testing.assert_array_equal(actual.obsm["spatial"], table.obsm["spatial"])
        assert actual.obs["spatialdata_instance_id"].tolist() == [0, 1, 2]
        assert isinstance(actual.obs["spatialdata_region"].dtype, pd.CategoricalDtype)
        declaration = actual.uns[POSITIONS_KEY]["coordinates"]["spatial"]
        assert declaration["units"] == "micrometer"
        assert declaration["frame"] == "global"
        assert declaration["calibration"]["global"]


class TestDirectReaderTableUse:
    @pytest.mark.parametrize("reader", ["curio", "calibrated_codex", "native_codex"])
    def test_table_access_copy_and_default_export_preserve_reader_output(self, tmp_path: Path, reader: str) -> None:
        if reader == "curio":
            write_curio(tmp_path)
            source = curio(tmp_path)
        else:
            write_csv(tmp_path / "cells.csv")
            source = codex(tmp_path, pixel_size_um=0.5 if reader == "calibrated_codex" else None)
        borrowed = source.tables["table"]
        original = borrowed.copy()
        assert borrowed is source.tables["table"]
        assert isinstance(borrowed.X, csr_array)
        assert isinstance(borrowed.obs, pd.DataFrame)
        borrowed.obs["analysis_group"] = pd.Categorical(["selected"] * borrowed.n_obs)
        assert "analysis_group" in source.tables["table"].obs
        owned = borrowed.copy()
        assert isinstance(owned.obs, pd.DataFrame)
        owned.obs["analysis_group"] = "changed"
        assert borrowed.obs["analysis_group"].tolist() == ["selected"] * borrowed.n_obs

        actual = to_legacy_anndata(source)

        assert_equal(actual, borrowed)
        assert actual.obs_names.tolist() == original.obs_names.tolist()
        assert actual.var_names.tolist() == original.var_names.tolist()
        assert_equal(actual.X, original.X)
        assert_equal(actual.layers, original.layers)
        if reader == "native_codex":
            assert "spatial" not in actual.obsm
            assert not actual.obsm
            assert isinstance(actual.obs, pd.DataFrame)
            assert isinstance(original.obs, pd.DataFrame)
            np.testing.assert_array_equal(actual.obs[["x", "y"]].to_numpy(), original.obs[["x", "y"]].to_numpy())
        else:
            np.testing.assert_array_equal(actual.obsm["spatial"], original.obsm["spatial"])
            declaration = actual.uns[POSITIONS_KEY]["coordinates"]["spatial"]
            assert declaration["units"] == "micrometer"
            assert declaration["calibration"]


class TestDocumentationExamples:
    def test_guide_examples_execute_on_their_documented_input_kinds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        guide = Path(__file__).parents[3] / "docs" / "legacy_anndata.md"
        snippets = re.findall(r"```python\n(.*?)```", guide.read_text(encoding="utf-8"), flags=re.DOTALL)
        assert snippets
        monkeypatch.chdir(tmp_path)
        for snippet in snippets:
            source = _linked()
            legacy = _table()
            del legacy.obsm["spatial_pixels"]
            del legacy.uns[POSITIONS_KEY]
            if 'library_id="sample_a"' in snippet:
                legacy.obs["library_id"] = pd.Categorical(["sample_a", "sample_b", "sample_a"])
                legacy.uns["spatial"] = {
                    "sample_a": {"metadata": {"source": "synthetic a"}},
                    "sample_b": {"metadata": {"source": "synthetic b"}},
                }
            table = source.tables["table"]
            if 'description["frame_key"]' in snippet:
                table.obs["frame"] = pd.Categorical(["a", "b", "a"])
                description = table.uns[POSITIONS_KEY]["coordinates"]["spatial"]
                description.pop("frame")
                description["frame_key"] = "frame"
                description["calibration"] = {"a": "field a calibration", "b": "field b calibration"}
            namespace: dict[str, object] = {
                "sdata": source,
                "adata": legacy,
                "table": table,
                "to_legacy_anndata": to_legacy_anndata,
                "from_legacy_anndata": from_legacy_anndata,
            }
            # Repository-owned executable examples are trusted source text.
            exec(compile(snippet, str(guide), "exec"), namespace)  # noqa: S102
        assert (tmp_path / "table.h5ad").is_file()
