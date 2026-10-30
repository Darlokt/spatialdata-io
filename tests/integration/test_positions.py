"""Executable persistence and upstream-behavior evidence for XY snapshots."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata._io.specs.registry import IORegistryError
from anndata.tests.helpers import assert_equal
from scipy import sparse
from spatialdata import SpatialData, read_zarr
from spatialdata.models import PointsModel, TableModel, get_table_keys
from spatialdata.transformations import Identity, Translation, get_transformation, set_transformation

from spatialdata_io._positions import POSITIONS_KEY, PositionsMetadata
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table

if TYPE_CHECKING:
    from pathlib import Path


def _table(*, row_frames: bool) -> ad.AnnData:
    native = np.asarray([[10, 20], [30, 40], [10, 20], [30, 40]], dtype=np.int64)
    physical = native.astype(np.float64) * 0.5
    regions = ["sample-a_cells"] * 2 + ["sample-b_cells"] * 2 if row_frames else ["cells"] * 4
    instances = [1, 2, 1, 2] if row_frames else [1, 2, 3, 4]
    obs = pd.DataFrame(
        {"region": pd.Categorical(regions), "instance": np.asarray(instances, dtype=np.int64)},
        index=pd.Index(["a1", "a2", "b1", "b2"]),
    )
    metadata: PositionsMetadata = {
        "schema_version": 1,
        "coordinates": {
            "spatial_grid": {
                "axes": ["x", "y"],
                "units": "grid",
                "frame": "source_grid",
                "position": "source_bin_reference_position",
            },
        },
    }
    if row_frames:
        obs["spatial_frame"] = pd.Categorical(
            ["sample-a", "sample-a", "sample-b", "sample-b"],
            categories=["sample-b", "sample-a"],
        )
        metadata["coordinates"]["spatial"] = {
            "axes": ["x", "y"],
            "units": "micrometer",
            "frame_key": "spatial_frame",
            "position": "source_bin_reference_position",
            "calibration": {
                "sample-a": "sample-a source resolution: 500 nm per grid unit",
                "sample-b": "sample-b source resolution: 500 nm per grid unit",
            },
        }
    else:
        metadata["coordinates"]["spatial"] = {
            "axes": ["x", "y"],
            "units": "micrometer",
            "frame": "global",
            "position": "source_bin_reference_position",
            "calibration": {"global": "source resolution: 500 nm per grid unit"},
        }
    expression = sparse.csr_array(np.asarray([[1, 0], [0, 2], [3, 0], [0, 4]], dtype=np.int32))
    table = ad.AnnData(expression, obs=obs, var=pd.DataFrame(index=["gene-b", "gene-a"]))
    table.layers["counts"] = expression.copy()
    table.layers["dense_annotation"] = np.arange(8, dtype=np.float32).reshape(4, 2)
    table.raw = ad.AnnData(
        sparse.csr_array(np.arange(12, dtype=np.int16).reshape(4, 3)),
        obs=pd.DataFrame(index=table.obs_names),
        var=pd.DataFrame(index=["raw-c", "raw-a", "raw-b"]),
    )
    table.obsm["spatial_grid"] = native
    table.obsm["spatial"] = physical
    # Unproved imported dimensions and integers retain their exact representation.
    table.obsm["spatial_source"] = np.asarray([[2**53 + 1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int64)
    table.uns[POSITIONS_KEY] = metadata
    expected = {region: pd.Index(obs.loc[obs["region"] == region, "instance"]) for region in dict.fromkeys(regions)}
    return parse_linked_table(
        table,
        TableLinkage(tuple(expected), "region", "instance"),
        context=ReaderErrorContext(reader="positions_contract", artifact="positions table"),
        expression_layers=("counts",),
        expected_instance_ids=expected,
    )


def _spatial_data(table: ad.AnnData, *, row_frames: bool) -> SpatialData:
    obs = table.obs
    xy_all = table.obsm["spatial"]
    assert isinstance(obs, pd.DataFrame)
    assert isinstance(xy_all, np.ndarray)
    points = {}
    for region in obs["region"].unique():
        selected = obs["region"] == region
        frame = str(obs.loc[selected, "spatial_frame"].iloc[0]) if row_frames else "global"
        xy = xy_all[selected.to_numpy()]
        positions = pd.DataFrame(xy, columns=["x", "y"], index=pd.Index(obs.loc[selected, "instance"]))
        points[str(region)] = PointsModel.parse(
            positions, coordinates={"x": "x", "y": "y"}, transformations={frame: Identity()}
        )
    return SpatialData(points=points, tables={"table": table})


class TestPositionsPersistence:
    @pytest.mark.parametrize("storage", ["h5ad", "anndata-zarr", "spatialdata-zarr"])
    @pytest.mark.parametrize("row_frames", [False, True])
    def test_arrays_metadata_linkage_and_expression_round_trip(
        self, tmp_path: Path, storage: Literal["h5ad", "anndata-zarr", "spatialdata-zarr"], *, row_frames: bool
    ) -> None:
        table = _table(row_frames=row_frames)
        before = table.copy()
        destination = tmp_path / storage
        # The external reader setting controls sparse container classes, not stored values.
        with ad.settings.override(use_sparse_array_on_read=True):
            if storage == "h5ad":
                table.write_h5ad(destination)
                restored = ad.read_h5ad(destination)
            elif storage == "anndata-zarr":
                table.write_zarr(destination)
                restored = ad.read_zarr(destination)
            else:
                _spatial_data(table, row_frames=row_frames).write(destination)
                restored = read_zarr(destination).tables["table"]

        assert_equal(table, before, exact=True)
        assert restored.obs_names.tolist() == ["a1", "a2", "b1", "b2"]
        assert restored.var_names.tolist() == ["gene-b", "gene-a"]
        assert restored.n_obs == table.n_obs
        actual_regions, actual_region_key, actual_instance_key = get_table_keys(restored)
        expected_regions, expected_region_key, expected_instance_key = get_table_keys(table)
        np.testing.assert_array_equal(actual_regions, expected_regions)
        assert (actual_region_key, actual_instance_key) == (expected_region_key, expected_instance_key)
        assert isinstance(restored.obs, pd.DataFrame)
        assert isinstance(table.obs, pd.DataFrame)
        pd.testing.assert_frame_equal(restored.obs, table.obs)
        for key in ("spatial", "spatial_grid", "spatial_source"):
            np.testing.assert_array_equal(restored.obsm[key], table.obsm[key], strict=True)
        physical = restored.obsm["spatial"]
        assert isinstance(physical, np.ndarray)
        assert physical.dtype == np.dtype(np.float64)
        assert np.isfinite(physical).all()
        assert isinstance(restored.X, sparse.csr_array)
        assert not isinstance(restored.X, sparse.spmatrix)
        assert_equal(restored.X, table.X, exact=True)
        for key in table.layers:
            assert_equal(restored.layers[key], table.layers[key], exact=True)
        assert_equal(restored.raw, table.raw, exact=True)
        actual = restored.uns[POSITIONS_KEY]
        assert actual["schema_version"] == 1
        assert set(actual["coordinates"]) == {"spatial", "spatial_grid"}
        for key, declaration in table.uns[POSITIONS_KEY]["coordinates"].items():
            loaded = actual["coordinates"][key]
            np.testing.assert_array_equal(loaded["axes"], ["x", "y"])
            assert set(loaded) == set(declaration)
            for field in set(declaration) - {"axes"}:
                assert loaded[field] == declaration[field]
        np.testing.assert_allclose(np.linalg.norm(physical[1] - physical[0]), np.sqrt(200))
        if row_frames:
            assert restored.obs["spatial_frame"].tolist() == ["sample-a", "sample-a", "sample-b", "sample-b"]
            assert restored.uns[POSITIONS_KEY]["coordinates"]["spatial"]["frame_key"] == "spatial_frame"
        else:
            assert "spatial_frame" not in restored.obs

    @pytest.mark.parametrize("storage", ["h5ad", "anndata-zarr"])
    @pytest.mark.parametrize("selection", ["reordered", "boolean", "empty"])
    def test_slices_keep_row_frames_and_surplus_calibration(self, tmp_path: Path, storage: str, selection: str) -> None:
        table = _table(row_frames=True)
        before = table.copy()
        selections: dict[str, list[int]] = {"reordered": [3, 0, 2], "boolean": [0, 1], "empty": []}
        indexes = selections[selection]
        selected = table[np.asarray([True, True, False, False])] if selection == "boolean" else table[indexes]
        selected = selected.copy()
        path = tmp_path / f"slice.{storage}"
        with ad.settings.override(use_sparse_array_on_read=True):
            if storage == "h5ad":
                selected.write_h5ad(path)
                restored = ad.read_h5ad(path)
            else:
                selected.write_zarr(path)
                restored = ad.read_zarr(path)
        assert restored.n_obs == len(indexes)
        assert restored.obs_names.tolist() == table.obs_names[indexes].tolist()
        assert isinstance(restored.obs, pd.DataFrame)
        assert isinstance(table.obs, pd.DataFrame)
        assert isinstance(restored.obs["spatial_frame"].dtype, pd.CategoricalDtype)
        expected_frames = table.obs["spatial_frame"].iloc[np.asarray(indexes, dtype=np.intp)].tolist()
        assert restored.obs["spatial_frame"].tolist() == expected_frames
        assert not restored.obs["spatial_frame"].isna().any()
        for key in table.obsm:
            np.testing.assert_array_equal(restored.obsm[key], table.obsm[key][indexes], strict=True)
        assert restored.obsm["spatial"].shape == (len(indexes), 2)
        assert restored.uns[POSITIONS_KEY]["coordinates"]["spatial"]["calibration"] == {
            "sample-a": "sample-a source resolution: 500 nm per grid unit",
            "sample-b": "sample-b source resolution: 500 nm per grid unit",
        }
        assert_equal(restored.X, selected.X, exact=True)
        if selection == "empty" and storage != "h5ad":
            # Installed AnnData Zarr drops an empty raw object; coordinates still persist.
            assert selected.raw is not None
            assert restored.raw is None
        else:
            assert_equal(restored.raw, selected.raw, exact=True)
        assert_equal(table, before, exact=True)

    def test_reordered_linked_slice_survives_spatialdata_zarr(self, tmp_path: Path) -> None:
        selected = _table(row_frames=True)[[3, 0, 2]].copy()
        path = tmp_path / "reordered.zarr"
        _spatial_data(selected, row_frames=True).write(path)
        with ad.settings.override(use_sparse_array_on_read=True):
            restored = read_zarr(path).tables["table"]
        assert restored.obs_names.tolist() == ["b2", "a1", "b1"]
        assert restored.obs["spatial_frame"].tolist() == ["sample-b", "sample-a", "sample-b"]
        for key in selected.obsm:
            np.testing.assert_array_equal(restored.obsm[key], selected.obsm[key], strict=True)
        assert_equal(restored.X, selected.X, exact=True)
        assert_equal(restored.raw, selected.raw, exact=True)


class TestPositionsExternalBehavior:
    @pytest.mark.parametrize("selection", [[0, 1], []])
    def test_slices_dropping_regions_need_explicit_linkage_update(self, selection: list[int]) -> None:
        table = _table(row_frames=True)
        selected = table[selection].copy()
        before = selected.copy()
        sdata = _spatial_data(table, row_frames=True)
        with pytest.raises(ValueError, match=r"Regions .* do not match"):
            sdata.tables["selected"] = selected
        assert "selected" not in sdata.tables
        assert selected.n_obs == len(selection)
        assert_equal(selected, before, exact=True)

    def test_tuple_axes_are_not_anndata_serializable(self, tmp_path: Path) -> None:
        table = _table(row_frames=False)
        table.uns[POSITIONS_KEY]["coordinates"]["spatial"]["axes"] = ("x", "y")
        with pytest.raises(IORegistryError, match="tuple"):
            table.write_zarr(tmp_path / "tuple-axes.zarr")

    def test_uns_is_not_observation_aligned(self) -> None:
        table = _table(row_frames=True)
        table.uns["incorrect_row_frames"] = np.asarray(["sample-a", "sample-a", "sample-b", "sample-b"])
        selected = table[[3, 0]]
        assert selected.obs["spatial_frame"].tolist() == ["sample-b", "sample-a"]
        np.testing.assert_array_equal(selected.uns["incorrect_row_frames"], table.uns["incorrect_row_frames"])
        assert len(selected.uns["incorrect_row_frames"]) != selected.n_obs

    def test_concat_defaults_drop_declaration_and_equal_globals_do_not_register_objects(self) -> None:
        first = _table(row_frames=False)
        second = _table(row_frames=False)
        original_first = first.copy()
        original_second = second.copy()
        plain = ad.concat({"first": first, "second": second}, index_unique="-")
        equal_uns = ad.concat({"first": first, "second": second}, uns_merge="same", index_unique="-")
        assert plain.n_obs == equal_uns.n_obs == 8
        assert POSITIONS_KEY not in plain.uns
        assert equal_uns.uns[POSITIONS_KEY] == first.uns[POSITIONS_KEY]
        assert equal_uns.uns[POSITIONS_KEY]["coordinates"]["spatial"]["frame"] == "global"
        assert "spatial_frame" not in equal_uns.obs
        np.testing.assert_array_equal(equal_uns.obsm["spatial"][:4], equal_uns.obsm["spatial"][4:])
        # Independent SpatialData objects both expose a local name, without registration.
        left = _spatial_data(first, row_frames=False)
        right = _spatial_data(second, row_frames=False)
        assert left.coordinate_systems == right.coordinate_systems == ["global"]
        assert left.points["cells"] is not right.points["cells"]
        assert_equal(first, original_first, exact=True)
        assert_equal(second, original_second, exact=True)

    def test_transform_change_does_not_refresh_snapshot(self) -> None:
        table = _table(row_frames=False)
        sdata = _spatial_data(table, row_frames=False)
        physical = table.obsm["spatial"]
        assert isinstance(physical, np.ndarray)
        snapshot = physical.copy()
        declaration = deepcopy(table.uns[POSITIONS_KEY])
        offset = Translation([100.0, -20.0], axes=("x", "y"))
        set_transformation(sdata.points["cells"], offset, to_coordinate_system="global")
        assert get_transformation(sdata.points["cells"], to_coordinate_system="global") == offset
        np.testing.assert_array_equal(table.obsm["spatial"], snapshot, strict=True)
        assert table.uns[POSITIONS_KEY] == declaration
        np.testing.assert_allclose(snapshot[0] + [100, -20], [105, -10])

    def test_table_model_accepts_nan_coordinates(self) -> None:
        table = ad.AnnData(sparse.csr_array([[1]], dtype=np.int16))
        table.obsm["spatial"] = np.asarray([[np.nan, 2]], dtype=np.float64)
        parsed = TableModel.parse(table)
        TableModel.validate(parsed)
        physical = parsed.obsm["spatial"]
        assert isinstance(physical, np.ndarray)
        assert np.isnan(physical[0, 0])
        assert parsed.n_obs == 1

    def test_explicit_copy_boundary_keeps_caller_coordinates_and_payload(self) -> None:
        caller = _table(row_frames=False)
        before = caller.copy()
        exported = caller.copy()
        exported_positions = exported.obsm["spatial"]
        assert isinstance(exported_positions, np.ndarray)
        exported_positions += np.asarray([100.0, 200.0])
        exported.uns[POSITIONS_KEY]["coordinates"]["spatial"]["frame"] = "registered_export"
        assert not np.shares_memory(exported.obsm["spatial"], caller.obsm["spatial"])
        assert_equal(caller, before, exact=True)
        for actual in (caller, exported):
            assert isinstance(actual.X, sparse.csr_array)
            assert_equal(actual.X, before.X, exact=True)
            assert_equal(actual.layers, before.layers, exact=True)
            assert_equal(actual.raw, before.raw, exact=True)
            assert actual.obs_names.tolist() == before.obs_names.tolist()
            assert actual.var_names.tolist() == before.var_names.tolist()
