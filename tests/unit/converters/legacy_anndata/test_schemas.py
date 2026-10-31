from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr
from anndata import AnnData, read_h5ad
from dask.callbacks import Callback
from scipy import sparse

from spatialdata_io._positions import POSITIONS_KEY, FixedFrame, PositionsMetadata
from spatialdata_io.converters.legacy_anndata._schemas import (
    coordinate_array,
    normalize_positions,
    observed_frames,
    parse_positions,
    physical_coordinates,
    validate_in_memory,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _table() -> AnnData:
    table = AnnData(sparse.csr_array([[1], [2]], dtype=np.int16))
    table.obsm["spatial"] = np.array([[1.0, 2.0], [3.0, 4.0]])
    return table


def _description() -> FixedFrame:
    return {
        "axes": ["x", "y"],
        "units": "micrometer",
        "position": "source cell centre",
        "frame": "sample",
        "calibration": {"sample": "verified source micrometres", "unused": "surplus calibration after slicing"},
    }


def _declared() -> AnnData:
    table = _table()
    metadata: PositionsMetadata = {"schema_version": 1, "coordinates": {"spatial": _description()}}
    table.uns[POSITIONS_KEY] = metadata
    return table


class TestValidateInMemory:
    def test_accepts_eager_slots_without_scanning_numerical_values(self, mocker: MockerFixture) -> None:
        table = _table()
        table.X = sparse.csr_matrix([[1], [2]], dtype=np.int16)
        table.layers["annotation"] = np.array([[np.nan], [np.inf]])
        table.obsm["strings"] = pd.DataFrame({"kind": ["a", "b"]}, index=table.obs_names)
        table.varm["metadata"] = np.array([["source"]])
        table.obsp["graph"] = sparse.csr_array(np.eye(2))
        table.varp["graph"] = np.ones((1, 1))
        table.raw = table
        finite = mocker.patch("numpy.isfinite", side_effect=AssertionError("expression must not be scanned"))

        validate_in_memory(table)

        finite.assert_not_called()

    def test_accepts_no_expression_and_view(self) -> None:
        table = AnnData(shape=(2, 1))
        validate_in_memory(table)
        validate_in_memory(table[:1])

    @pytest.mark.parametrize("slot", ["X", "layers", "obsm", "varm", "obsp", "varp", "raw.X", "raw.varm"])
    def test_rejects_lazy_slots_without_executing_tasks(self, slot: str) -> None:
        table = _table()
        if slot == "X":
            table.X = da.ones((2, 1), chunks=(1, 1))
        elif slot == "layers":
            table.layers["lazy"] = da.ones((2, 1), chunks=(1, 1))
        elif slot == "obsm":
            table.obsm["lazy"] = da.ones((2, 2), chunks=(1, 2))
        elif slot == "varm":
            table.varm["lazy"] = da.ones((1, 2), chunks=(1, 2))
        elif slot == "obsp":
            table.obsp["lazy"] = da.ones((2, 2), chunks=(1, 2))
        elif slot == "varp":
            table.varp["lazy"] = da.ones((1, 1), chunks=(1, 1))
        elif slot == "raw.X":
            source = _table()
            source.X = da.ones((2, 1), chunks=(1, 1))
            table.raw = source
        else:
            source = _table()
            source.varm["lazy"] = da.ones((1, 2), chunks=(1, 2))
            table.raw = source
        tasks: list[str] = []
        with (
            Callback(pretask=lambda key, _graph, _state: tasks.append(str(key))),
            pytest.raises(TypeError, match="eager NumPy, pandas or SciPy"),
        ):
            validate_in_memory(table)
        assert tasks == []

    @pytest.mark.parametrize("slot", ["obs", "var", "raw.var"])
    def test_rejects_xarray_annotations(self, slot: str) -> None:
        table = _table()
        if slot == "obs":
            table.obs = xr.Dataset({"kind": ("cell", ["a", "b"])}, coords={"cell": table.obs_names})
        elif slot == "var":
            table.var = xr.Dataset({"kind": ("feature", ["a"])}, coords={"feature": table.var_names})
        else:
            source = _table()
            source.var = xr.Dataset({"kind": ("feature", ["a"])}, coords={"feature": source.var_names})
            table.raw = source
        with pytest.raises(TypeError, match="pandas DataFrame"):
            validate_in_memory(table)

    def test_rejects_backed_before_accessing_expression(self, tmp_path: Path) -> None:
        path = tmp_path / "table.h5ad"
        _table().write_h5ad(path)
        table = read_h5ad(path, backed="r")
        try:
            with pytest.raises(ValueError, match="backed"):
                validate_in_memory(table)
        finally:
            table.file.close()


class TestCoordinateArray:
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float32, np.float64])
    def test_borrows_valid_array_without_dtype_conversion(self, dtype: npt.DTypeLike) -> None:
        table = _table()
        values = np.array([[2**53 + 1, 2], [3, 4]], dtype=dtype)
        table.obsm["native"] = values
        assert coordinate_array(table, "native") is values

    def test_accepts_empty_array(self) -> None:
        table = AnnData(shape=(0, 1))
        values = np.empty((0, 2), dtype=np.int64)
        table.obsm["native"] = values
        assert coordinate_array(table, "native") is values

    def test_rejects_missing_embedding(self) -> None:
        with pytest.raises(ValueError, match="missing obsm"):
            coordinate_array(_table(), "missing")

    @pytest.mark.parametrize("dtype", [np.bool_, np.complex128, object, "U1"])
    def test_rejects_nonreal_dtypes(self, dtype: npt.DTypeLike) -> None:
        table = _table()
        table.obsm["invalid"] = np.ones((2, 2), dtype=dtype)
        with pytest.raises(TypeError, match="real numerical NumPy"):
            coordinate_array(table, "invalid")

    def test_rejects_dataframe_coordinates(self) -> None:
        table = _table()
        table.obsm["invalid"] = pd.DataFrame({"x": [1, 2], "y": [3, 4]}, index=table.obs_names)
        with pytest.raises(TypeError, match="real numerical NumPy"):
            coordinate_array(table, "invalid")

    @pytest.mark.parametrize("shape", [(2, 1), (2, 3), (2, 2, 1)])
    def test_rejects_nonxy_shape(self, shape: tuple[int, ...]) -> None:
        table = _table()
        table.obsm["invalid"] = np.zeros(shape)
        with pytest.raises(ValueError, match="XY coordinates"):
            coordinate_array(table, "invalid")

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite_coordinates(self, value: float) -> None:
        table = _table()
        table.obsm["spatial"][1, 0] = value
        with pytest.raises(ValueError, match="nonfinite"):
            coordinate_array(table, "spatial")


class TestPhysicalCoordinates:
    @pytest.mark.parametrize(
        "dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint64, np.float16, np.float32]
    )
    def test_lossless_conversion(self, dtype: npt.DTypeLike) -> None:
        values = np.array([[1, 2], [3, 4]], dtype=dtype)
        actual = physical_coordinates(values, key="spatial")
        assert actual.dtype == np.float64
        np.testing.assert_array_equal(actual, values)

    def test_borrows_existing_float64(self) -> None:
        values = np.array([[0.25, 2.5], [3.0, 4.0]])
        assert physical_coordinates(values, key="spatial") is values

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_preserves_representable_large_integers(self, dtype: npt.DTypeLike) -> None:
        values = np.array([[2**53, 2**53 + 2], [2**60, 2**63]], dtype=np.uint64)
        if dtype == np.int64:
            values = np.array([[2**53, 2**53 + 2], [-(2**60), -(2**63)]], dtype=dtype)
        actual = physical_coordinates(values, key="spatial")
        assert all(int(expected) == int(result) for expected, result in zip(values.flat, actual.flat, strict=True))

    @pytest.mark.parametrize("value", [2**53 + 1, 2**63 - 1, 2**64 - 1])
    def test_rejects_lossy_integer_conversion(self, value: int) -> None:
        values = np.array([[value, 0]], dtype=np.uint64)
        with pytest.raises(ValueError, match="losslessly"):
            physical_coordinates(values, key="spatial")

    def test_rejects_lossy_wide_float_conversion(self) -> None:
        if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
            pytest.skip("This platform does not provide a floating dtype wider than float64.")
        values = np.array([[np.longdouble(1) + np.finfo(np.longdouble).eps, 0]], dtype=np.longdouble)
        with pytest.raises(ValueError, match="losslessly"):
            physical_coordinates(values, key="spatial")


class TestParsePositions:
    def test_absent_metadata_stays_absent(self) -> None:
        table = _table()
        assert parse_positions(table) == {"schema_version": 1, "coordinates": {}}
        assert POSITIONS_KEY not in table.uns

    def test_normalizes_persisted_metadata_without_mutating_caller(self) -> None:
        table = _declared()
        table.uns[POSITIONS_KEY]["schema_version"] = np.int64(1)
        axes = np.array(["x", "y"])
        table.uns[POSITIONS_KEY]["coordinates"]["spatial"]["axes"] = axes

        actual = parse_positions(table)

        assert actual == {"schema_version": 1, "coordinates": {"spatial": _description()}}
        assert type(actual["schema_version"]) is int
        assert actual["coordinates"]["spatial"]["axes"] == ["x", "y"]
        assert table.uns[POSITIONS_KEY]["coordinates"]["spatial"]["axes"] is axes
        actual["coordinates"]["spatial"]["calibration"]["sample"] = "modified output"
        assert table.uns[POSITIONS_KEY]["coordinates"]["spatial"]["calibration"]["sample"] != "modified output"

    @pytest.mark.parametrize("units", ["pixel", "grid", "unknown"])
    def test_accepts_native_declared_units(self, units: str) -> None:
        table = _declared()
        description = table.uns[POSITIONS_KEY]["coordinates"]["spatial"]
        description["units"] = units
        description.pop("calibration")
        table.obsm["spatial"] = np.array([[2**53 + 1, 0], [3, 4]], dtype=np.uint64)
        assert parse_positions(table)["coordinates"]["spatial"]["units"] == units

    def test_validates_observed_row_frames_and_allows_unused_calibrations(self) -> None:
        table = _declared()
        table.obs["frames"] = pd.Categorical(["b", "a"], categories=["a", "b", "unused"])
        description = table.uns[POSITIONS_KEY]["coordinates"]["spatial"]
        description.pop("frame")
        description["frame_key"] = "frames"
        description["calibration"] = {"a": "calibration a", "b": "calibration b", "extra": "sliced away"}
        metadata = parse_positions(table)
        assert observed_frames(table, metadata["coordinates"]["spatial"]) == ("b", "a")

    def test_accepts_empty_physical_fixed_frame(self) -> None:
        table = _declared()[:0].copy()
        actual = parse_positions(table)
        assert observed_frames(table, actual["coordinates"]["spatial"]) == ("sample",)

    def test_accepts_empty_physical_row_frames_without_inventing_a_frame(self) -> None:
        table = _declared()[:0].copy()
        table.obs["frames"] = pd.Categorical([], categories=["sample"])
        description = table.uns[POSITIONS_KEY]["coordinates"]["spatial"]
        description.pop("frame")
        description["frame_key"] = "frames"
        metadata = parse_positions(table)
        assert observed_frames(table, metadata["coordinates"]["spatial"]) == ()
        assert "frame" not in metadata["coordinates"]["spatial"]

    @pytest.mark.parametrize("value", [True, np.bool_(1), 0, 2, "1", 1.0, None])
    def test_rejects_unsupported_versions(self, value: object) -> None:
        table = _declared()
        table.uns[POSITIONS_KEY]["schema_version"] = value
        with pytest.raises(ValueError, match="schema_version 1"):
            parse_positions(table)

    @pytest.mark.parametrize("value", [[], None, {1: "nonstring key"}])
    def test_rejects_nonmapping_metadata(self, value: object) -> None:
        table = _table()
        table.uns[POSITIONS_KEY] = value
        with pytest.raises(TypeError, match="mapping with string keys"):
            parse_positions(table)

    @pytest.mark.parametrize("key", ["coordinates", "schema_version"])
    def test_rejects_missing_top_level_fields(self, key: str) -> None:
        table = _declared()
        table.uns[POSITIONS_KEY].pop(key)
        with pytest.raises(ValueError, match="missing fields"):
            parse_positions(table)

    def test_rejects_unknown_top_level_fields(self) -> None:
        table = _declared()
        table.uns[POSITIONS_KEY]["unexpected"] = "unrecognized meaning"
        with pytest.raises(ValueError, match="unsupported fields"):
            parse_positions(table)

    def test_rejects_empty_coordinate_key(self) -> None:
        table = _declared()
        table.uns[POSITIONS_KEY]["coordinates"][""] = _description()
        with pytest.raises(ValueError, match="nonempty strings"):
            parse_positions(table)

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("axes", ["y", "x"], "XY order"),
            ("axes", ["x"], "XY order"),
            ("axes", ["x", 1], "XY order"),
            ("axes", np.array([["x", "y"]]), "XY order"),
            ("axes", ("x", "y"), "XY order"),
            ("units", "nanometer", "units must"),
            ("units", 1, "units must"),
            ("position", "", "nonempty string"),
            ("position", None, "nonempty string"),
            ("frame", " ", "nonempty string"),
            ("calibration", {"sample": ""}, "nonempty string"),
            ("calibration", {"": "reason"}, "nonempty string"),
        ],
    )
    def test_rejects_malformed_description_fields(self, key: str, value: object, message: str) -> None:
        table = _declared()
        table.uns[POSITIONS_KEY]["coordinates"]["spatial"][key] = value
        with pytest.raises(ValueError, match=message):
            parse_positions(table)

    @pytest.mark.parametrize("key", ["axes", "units", "position"])
    def test_rejects_missing_description_fields(self, key: str) -> None:
        table = _declared()
        table.uns[POSITIONS_KEY]["coordinates"]["spatial"].pop(key)
        with pytest.raises(ValueError, match="missing fields"):
            parse_positions(table)

    def test_rejects_unknown_description_fields(self) -> None:
        table = _declared()
        table.uns[POSITIONS_KEY]["coordinates"]["spatial"]["scale"] = 5
        with pytest.raises(ValueError, match="unsupported fields"):
            parse_positions(table)

    @pytest.mark.parametrize("both", [False, True])
    def test_requires_one_frame_definition(self, *, both: bool) -> None:
        table = _declared()
        description = table.uns[POSITIONS_KEY]["coordinates"]["spatial"]
        if both:
            description["frame_key"] = "frames"
        else:
            description.pop("frame")
        with pytest.raises(ValueError, match="exactly one"):
            parse_positions(table)

    def test_rejects_missing_frame_column(self) -> None:
        table = _declared()
        description = table.uns[POSITIONS_KEY]["coordinates"]["spatial"]
        description.pop("frame")
        description["frame_key"] = "frames"
        with pytest.raises(ValueError, match="missing obs"):
            parse_positions(table)

    def test_rejects_ambiguous_duplicate_frame_columns(self) -> None:
        table = _declared()
        table.obs = pd.DataFrame(
            {"first": pd.Categorical(["sample", "sample"]), "second": pd.Categorical(["sample", "sample"])},
            index=table.obs_names,
        )
        table.obs.columns = ["frames", "frames"]
        description = table.uns[POSITIONS_KEY]["coordinates"]["spatial"]
        description.pop("frame")
        description["frame_key"] = "frames"
        with pytest.raises(ValueError, match="categorical string"):
            parse_positions(table)

    @pytest.mark.parametrize(
        "frames",
        [
            pd.Series(["a", "b"]),
            pd.Series(pd.Categorical(["a", None])),
            pd.Series(pd.Categorical([1, 2])),
            pd.Series(pd.Categorical(["a", "b"], categories=["a", "b", " "])),
        ],
    )
    def test_rejects_invalid_frame_columns(self, frames: pd.Series) -> None:
        table = _declared()
        table.obs["frames"] = frames.array
        description = table.uns[POSITIONS_KEY]["coordinates"]["spatial"]
        description.pop("frame")
        description["frame_key"] = "frames"
        with pytest.raises(ValueError, match="categorical string"):
            parse_positions(table)

    def test_rejects_lazy_frame_annotations(self) -> None:
        table = _declared()
        table.obs = xr.Dataset({"frame": ("cell", ["a", "b"])}, coords={"cell": table.obs_names})
        description = table.uns[POSITIONS_KEY]["coordinates"]["spatial"]
        description.pop("frame")
        description["frame_key"] = "frame"
        with pytest.raises(TypeError, match="pandas observation"):
            parse_positions(table)

    @pytest.mark.parametrize("calibration", [{}, {"unrelated": "calibrated elsewhere"}, None])
    def test_requires_physical_calibration(self, calibration: dict[str, str] | None) -> None:
        table = _declared()
        description = table.uns[POSITIONS_KEY]["coordinates"]["spatial"]
        if calibration is None:
            description.pop("calibration")
        else:
            description["calibration"] = calibration
        with pytest.raises(ValueError, match="requires calibration"):
            parse_positions(table)


class TestNormalizePositions:
    def test_preserves_undeclared_arbitrary_source_without_claims(self) -> None:
        table = _table()
        source = np.array([[2**53 + 1, 2, 7], [3, 4, 9]], dtype=np.uint64)
        table.obsm["spatial"] = source
        metadata = normalize_positions(table, parse_positions(table))
        assert "spatial" not in table.obsm
        assert table.obsm["spatial_source"] is source
        assert POSITIONS_KEY not in table.uns
        assert metadata == {"schema_version": 1, "coordinates": {}}

    @pytest.mark.parametrize(
        ("units", "destination"), [("pixel", "spatial_pixels"), ("grid", "spatial_grid"), ("unknown", "spatial_source")]
    )
    def test_moves_complete_native_declaration(self, units: str, destination: str) -> None:
        table = _declared()
        table.uns[POSITIONS_KEY]["coordinates"]["spatial"]["units"] = units
        source = np.array([[2**53 + 1, 2], [3, 4]], dtype=np.int64)
        table.obsm["spatial"] = source
        metadata = parse_positions(table)
        description = metadata["coordinates"]["spatial"]

        actual = normalize_positions(table, metadata)

        assert table.obsm[destination] is source
        assert "spatial" not in table.obsm
        assert actual["coordinates"][destination] is description
        assert "spatial" not in actual["coordinates"]
        assert table.uns[POSITIONS_KEY] is actual

    @pytest.mark.parametrize("declared", [False, True])
    def test_rejects_occupied_preservation_key_without_partial_moves(self, *, declared: bool) -> None:
        table = _declared() if declared else _table()
        if declared:
            table.uns[POSITIONS_KEY]["coordinates"]["spatial"]["units"] = "unknown"
        source = table.obsm["spatial"]
        table.obsm["spatial_source"] = np.zeros((2, 3))
        metadata = parse_positions(table)
        with pytest.raises(ValueError, match="occupied"):
            normalize_positions(table, metadata)
        assert table.obsm["spatial"] is source

    def test_normalizes_physical_arrays_and_preserves_native_values(self) -> None:
        table = _declared()
        physical = np.array([[0.25, 2.0], [3.5, 4.0]], dtype=np.float32)
        native = np.array([[2**53 + 1, 2], [3, 4]], dtype=np.int64)
        table.obsm["spatial"] = physical
        table.obsm["spatial_pixels"] = native
        table.uns[POSITIONS_KEY]["coordinates"]["spatial_pixels"] = {
            "axes": ["x", "y"],
            "units": "pixel",
            "position": "source pixel centre",
            "frame": "native",
        }

        metadata = normalize_positions(table, parse_positions(table))

        assert table.obsm["spatial"].dtype == np.float64
        np.testing.assert_array_equal(table.obsm["spatial"], physical)
        assert table.obsm["spatial_pixels"] is native
        assert table.uns[POSITIONS_KEY] is metadata

    def test_does_not_promote_other_physical_embeddings(self) -> None:
        table = _declared()
        table.obsm["physical"] = table.obsm.pop("spatial")
        descriptions = table.uns[POSITIONS_KEY]["coordinates"]
        descriptions["physical"] = descriptions.pop("spatial")
        normalize_positions(table, parse_positions(table))
        assert "spatial" not in table.obsm
        assert "physical" in table.obsm

    def test_removes_empty_metadata(self) -> None:
        table = AnnData(shape=(0, 1))
        metadata: PositionsMetadata = {"schema_version": 1, "coordinates": {}}
        table.uns[POSITIONS_KEY] = metadata
        normalize_positions(table, parse_positions(table))
        assert POSITIONS_KEY not in table.uns

    def test_rejects_lossy_physical_input(self) -> None:
        table = _declared()
        table.obsm["spatial"] = np.array([[2**53 + 1, 0], [3, 4]], dtype=np.int64)
        with pytest.raises(ValueError, match="losslessly"):
            normalize_positions(table, parse_positions(table))
