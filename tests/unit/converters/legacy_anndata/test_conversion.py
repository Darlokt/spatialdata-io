"""Check public selection, calibration and mutation boundaries."""

from __future__ import annotations

import inspect
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from anndata.tests.helpers import assert_equal
from scipy.sparse import csr_array, csr_matrix
from spatialdata import SpatialData
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.converters.legacy_anndata import from_legacy_anndata, to_legacy_anndata
from spatialdata_io.experimental import from_legacy_anndata as experimental_import
from spatialdata_io.experimental import to_legacy_anndata as experimental_export


def _table(*, units: Literal["micrometer", "pixel", "grid", "unknown"] = "micrometer") -> AnnData:
    table = AnnData(csr_array([[1], [2]], dtype=np.int16), obs=pd.DataFrame(index=["first", "second"]))
    table.obsm["spatial"] = np.asarray([[10.0, 20.0], [30.0, 40.0]])
    description: dict[str, object] = {
        "axes": ["x", "y"],
        "units": units,
        "frame": "sample",
        "position": "vendor_centre",
    }
    if units == "micrometer":
        description["calibration"] = {"sample": "verified acquisition calibration"}
    table.uns[POSITIONS_KEY] = {"schema_version": 1, "coordinates": {"spatial": description}}
    return table


def _linked(*, units: Literal["micrometer", "pixel", "grid", "unknown"] = "micrometer") -> SpatialData:
    table = _table(units=units)
    table.obs["region"] = pd.Categorical(["cells", "cells"])
    table.obs["id"] = [11, 8]
    table.uns["spatialdata_attrs"] = {"region": "cells", "region_key": "region", "instance_key": "id"}
    shapes = ShapesModel.parse(
        np.asarray([[2.0, 3.0], [4.0, 5.0]]),
        geometry=0,
        radius=1.0,
        index=pd.Index([8, 11]),
        transformations={"sample": Identity()},
    )
    return SpatialData(tables={"table": table}, shapes={"cells": shapes})


class TestToLegacyAnndata:
    @pytest.mark.parametrize("names", [(), ("one", "two")])
    def test_requires_unambiguous_table(self, names: tuple[str, ...]) -> None:
        sdata = SpatialData(tables={name: _table() for name in names})
        with pytest.raises(ValueError, match="table_name"):
            to_legacy_anndata(sdata)

    @pytest.mark.parametrize("name", ["", "missing"])
    def test_rejects_unknown_table(self, name: str) -> None:
        with pytest.raises(ValueError, match="Unknown table_name"):
            to_legacy_anndata(SpatialData(tables={"table": _table()}), table_name=name)

    @pytest.mark.parametrize("frame", ["", 3, False])
    def test_rejects_invalid_frame_option(self, frame: str | int) -> None:
        with pytest.raises(TypeError, match="coordinate_system"):
            to_legacy_anndata(SpatialData(tables={"table": _table()}), coordinate_system=frame)  # pyrefly: ignore[bad-argument-type] - Exercise untrusted public options.

    @pytest.mark.parametrize("mode", ["", "geometry", True])
    def test_rejects_unknown_position_source(self, *, mode: str | bool) -> None:
        with pytest.raises(ValueError, match="position_source"):
            to_legacy_anndata(SpatialData(tables={"table": _table()}), position_source=mode)  # pyrefly: ignore[bad-argument-type] - Exercise untrusted public options.

    def test_rejects_unestablished_frame(self) -> None:
        with pytest.raises(ValueError, match="No table position declaration"):
            to_legacy_anndata(SpatialData(tables={"table": _table()}), coordinate_system="unknown")

    def test_rejects_conflicting_units_for_same_frame(self) -> None:
        table = _table()
        table.obsm["native"] = np.ones((2, 2))
        table.uns[POSITIONS_KEY]["coordinates"]["native"] = {
            "axes": ["x", "y"],
            "units": "pixel",
            "frame": "sample",
            "position": "native_index",
        }
        with pytest.raises(ValueError, match="Conflicting position units"):
            to_legacy_anndata(SpatialData(tables={"table": table}))

    def test_selects_explicit_native_frame_while_preserving_physical_positions(self) -> None:
        table = _table()
        table.obsm["spatial_grid"] = np.ones((2, 2), dtype=np.int16)
        table.uns[POSITIONS_KEY]["coordinates"]["spatial_grid"] = {
            "axes": ["x", "y"],
            "units": "grid",
            "frame": "capture",
            "position": "capture_grid",
        }
        exported = to_legacy_anndata(SpatialData(tables={"table": table}), coordinate_system="capture")
        assert_equal(exported, table)

    def test_requires_established_units_for_centroids(self) -> None:
        sdata = _linked(units="unknown")
        with pytest.raises(ValueError, match="units are unknown"):
            to_legacy_anndata(sdata, position_source="centroids")

    def test_does_not_infer_centroid_frame_from_global_name(self) -> None:
        sdata = _linked()
        del sdata.tables["table"].uns[POSITIONS_KEY]
        with pytest.raises(ValueError, match="established coordinate frame"):
            to_legacy_anndata(sdata, position_source="centroids")

    @pytest.mark.parametrize(
        ("units", "destination"), [("micrometer", "spatial_original"), ("pixel", "spatial_centroids")]
    )
    def test_rejects_occupied_centroid_destinations(
        self, units: Literal["micrometer", "pixel"], destination: str
    ) -> None:
        sdata = _linked(units=units)
        sdata.tables["table"].obsm[destination] = np.zeros((2, 2))
        before = sdata.tables["table"].copy()
        with pytest.raises(ValueError, match="occupied"):
            to_legacy_anndata(sdata, position_source="centroids")
        assert_equal(sdata.tables["table"], before)

    def test_promotes_verified_alternative_physical_embedding_only_when_centroids_requested(self) -> None:
        sdata = _linked()
        table = sdata.tables["table"]
        table.obsm["verified_centres"] = table.obsm.pop("spatial")
        table.uns[POSITIONS_KEY]["coordinates"]["verified_centres"] = table.uns[POSITIONS_KEY]["coordinates"].pop(
            "spatial"
        )
        exported = to_legacy_anndata(sdata, position_source="centroids")
        np.testing.assert_array_equal(exported.obsm["spatial"], [[4.0, 5.0], [2.0, 3.0]])
        assert "spatial_original" not in exported.obsm
        np.testing.assert_array_equal(exported.obsm["verified_centres"], [[10.0, 20.0], [30.0, 40.0]])

    def test_positionless_output_is_owned_and_sparse(self) -> None:
        table = AnnData(csr_matrix([[2], [7]], dtype=np.int32))
        output = to_legacy_anndata(SpatialData(tables={"table": table}))
        assert "spatial" not in output.obsm
        assert isinstance(output.X, csr_array)
        output.X.data[0] = 99
        assert isinstance(table.X, csr_matrix)
        assert table.X.data[0] == 2


class TestFromLegacyAnndata:
    @pytest.mark.parametrize("rgb", [0, "rgb", np.bool_(1)])
    def test_rejects_invalid_rgb_option(self, rgb: int | str | np.bool_) -> None:
        with pytest.raises(TypeError, match="rgb"):
            from_legacy_anndata(_table(), rgb=rgb)  # pyrefly: ignore[bad-argument-type] - Exercise untrusted RGB options.

    @pytest.mark.parametrize("units", ["grid", "unknown"])
    def test_rejects_unsupported_declared_legacy_units(self, units: Literal["grid", "unknown"]) -> None:
        with pytest.raises(ValueError, match="pixel or micrometer"):
            from_legacy_anndata(_table(units=units))

    def test_rejects_legacy_scale_as_physical_calibration(self) -> None:
        table = _table()
        table.uns["spatial"] = {
            "sample": {
                "images": {"hires": np.zeros((2, 3, 3), dtype=np.uint8)},
                "scalefactors": {"tissue_hires_scalef": 1.0},
            }
        }
        with pytest.raises(ValueError, match="physical image mapping"):
            from_legacy_anndata(table)

    @pytest.mark.parametrize("key", ["spatialdata_region", "spatialdata_instance_id"])
    def test_rejects_occupied_generated_linkage_columns(self, key: str) -> None:
        table = _table()
        table.obs[key] = [1, 2]
        with pytest.raises(ValueError, match="occupied"):
            from_legacy_anndata(table)

    @pytest.mark.parametrize("units", ["micrometer", "pixel"])
    def test_empty_row_frame_positions_remain_unlinked(self, units: Literal["micrometer", "pixel"]) -> None:
        table = _table(units=units)[:0].copy()
        description = table.uns[POSITIONS_KEY]["coordinates"]["spatial"]
        del description["frame"]
        description["frame_key"] = "sample_frame"
        table.obs["sample_frame"] = pd.Categorical([], categories=["sample"])
        imported = from_legacy_anndata(table)
        key = "spatial" if units == "micrometer" else "spatial_pixels"
        assert imported.tables["table"].obsm[key].shape == (0, 2)
        assert not imported.points
        assert "spatialdata_attrs" not in imported.tables["table"].uns
        exported = to_legacy_anndata(imported)
        assert exported.obsm[key].shape == (0, 2)

    def test_empty_row_frame_cannot_assign_an_image_to_an_unknown_frame(self) -> None:
        table = _table(units="pixel")[:0].copy()
        description = table.uns[POSITIONS_KEY]["coordinates"]["spatial"]
        del description["frame"]
        description["frame_key"] = "sample_frame"
        table.obs["sample_frame"] = pd.Categorical([], categories=["sample"])
        table.uns["spatial"] = {
            "sample": {"images": {"hires": np.zeros((2, 3, 3), dtype=np.uint8)}, "scalefactors": {"hires": 1.0}}
        }
        with pytest.raises(ValueError, match="do not establish an image frame"):
            from_legacy_anndata(table)

    def test_native_only_and_positionless_inputs_clear_stale_linkage(self) -> None:
        table = _linked().tables["table"]
        table.obsm["spatial_pixels"] = table.obsm.pop("spatial")
        table.uns[POSITIONS_KEY]["coordinates"]["spatial_pixels"] = table.uns[POSITIONS_KEY]["coordinates"].pop(
            "spatial"
        )
        result = from_legacy_anndata(table)
        assert "spatialdata_attrs" not in result.tables["table"].uns
        assert not result.points
        assert "spatialdata_attrs" in table.uns


class TestPackage:
    def test_exports_and_keyword_only_signature(self) -> None:
        assert experimental_import is from_legacy_anndata
        assert experimental_export is to_legacy_anndata
        signature = inspect.signature(to_legacy_anndata)
        assert tuple(signature.parameters) == ("sdata", "table_name", "coordinate_system", "position_source")
        assert signature.parameters["position_source"].default == "table"
        assert all(
            value.kind == inspect.Parameter.KEYWORD_ONLY
            for key, value in signature.parameters.items()
            if key != "sdata"
        )
