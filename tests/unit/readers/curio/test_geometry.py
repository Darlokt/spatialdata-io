"""Curio bead coordinate validation and float64 geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.curio._geometry import read_shapes
from spatialdata_io.readers.curio._layout import discover_layout
from tests.integration.readers.curio._fixtures import write_curio

if TYPE_CHECKING:
    from pathlib import Path


class TestReadShapes:
    """Reject invalid coordinates before constructing a spatial element."""

    def test_finite_extended_precision_coordinates_cannot_overflow_output(self, tmp_path: Path) -> None:
        if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
            pytest.skip("Extended precision has the same range as float64.")
        write_curio(tmp_path, duplicate_coordinates=False)
        layout = discover_layout(tmp_path)
        table = ad.read_h5ad(layout.counts)
        native = np.full((3, 2), np.longdouble(np.finfo(np.float64).max) * 2)
        table.obsm["X_spatial"] = native

        with pytest.raises(ReaderFormatError, match="overflow"):
            read_shapes(table, layout.counts)

        assert table.obsm["X_spatial"] is native
        assert "spatial_pixels" not in table.obsm
        assert "spatial" not in table.obsm

    def test_large_integer_coordinates_round_in_float64_output(self, tmp_path: Path) -> None:
        write_curio(tmp_path, duplicate_coordinates=False)
        layout = discover_layout(tmp_path)
        table = ad.read_h5ad(layout.counts)
        native = np.full((3, 2), 2**53 + 1, dtype=np.int64)
        table.obsm["X_spatial"] = native

        shapes = read_shapes(table, layout.counts)

        np.testing.assert_array_equal(shapes.geometry.x, np.full(3, float(2**53)))
        np.testing.assert_array_equal(table.obsm["spatial"], native.astype(np.float64) * 0.647)
        assert table.obsm["spatial_pixels"] is native

    @pytest.mark.parametrize("kind", ["missing_duplicate", "shape", "dtype", "nonfinite", "disagree"])
    def test_invalid_spatial_arrays(self, tmp_path: Path, kind: str) -> None:
        write_curio(tmp_path)
        layout = discover_layout(tmp_path)
        table = ad.read_h5ad(layout.counts)
        if kind == "missing_duplicate":
            del table.obsm["X_spatial"]
        elif kind == "shape":
            table.obsm["spatial"] = np.ones((3, 3))
        elif kind == "dtype":
            table.obsm["spatial"] = np.full((3, 2), "x")
        elif kind == "nonfinite":
            table.obsm["spatial"] = np.full((3, 2), np.nan)
        else:
            table.obsm["X_spatial"] = np.ones((3, 2))

        with pytest.raises(ReaderFormatError, match="spatial"):
            read_shapes(table, layout.counts)

    @pytest.mark.parametrize("destination", ["spatial_pixels", "spatialdata_io_positions", "X_SPATIAL"])
    def test_reserved_destinations_and_unsupported_profile(self, tmp_path: Path, destination: str) -> None:
        write_curio(tmp_path)
        layout = discover_layout(tmp_path)
        table = ad.read_h5ad(layout.counts)
        if destination == "spatialdata_io_positions":
            table.uns[destination] = {"source": "retained"}
        else:
            table.obsm[destination] = np.ones((3, 2))
        with pytest.raises(ReaderFormatError, match=r"destination|profile"):
            read_shapes(table, layout.counts)

    def test_native_buffer_and_expression_ownership(self, tmp_path: Path) -> None:
        write_curio(tmp_path, duplicate_coordinates=False)
        layout = discover_layout(tmp_path)
        table = ad.read_h5ad(layout.counts)
        native = table.obsm["X_spatial"]
        expression = table.X
        read_shapes(table, layout.counts)
        assert table.obsm["spatial_pixels"] is native
        assert table.X is expression
        np.testing.assert_array_equal(native, [[30, 4], [10, 8], [20, 2]])
        assert table.obsm["spatial"].dtype == np.float64
        assert table.obsm["spatial"] is not native

    def test_shape_index_does_not_alias_observation_names(self, tmp_path: Path) -> None:
        """Renaming the returned index must not rename the table's observation index."""
        write_curio(tmp_path)
        layout = discover_layout(tmp_path)
        table = ad.read_h5ad(layout.counts)

        shapes = read_shapes(table, layout.counts)
        shapes.index.name = "bead"

        assert shapes.index.tolist() == table.obs_names.tolist()
        assert table.obs_names.name is None

    @pytest.mark.parametrize(
        "inputs",
        [("int32", "float64", -30.0), ("int64", "uint64", 30.0), ("float32", "float64", -30.5)],
    )
    @pytest.mark.parametrize("reverse", [False, True])
    @pytest.mark.parametrize("agree", [False, True])
    def test_mixed_numeric_coordinates(
        self,
        tmp_path: Path,
        inputs: tuple[str, str, float],
        *,
        reverse: bool,
        agree: bool,
    ) -> None:
        write_curio(tmp_path)
        layout = discover_layout(tmp_path)
        table = ad.read_h5ad(layout.counts)
        first_dtype, second_dtype, value = inputs
        first = np.asarray([[value, 4], [10, 8], [20, 2]], dtype=first_dtype)
        second = first.astype(second_dtype)
        if not agree:
            second[0, 1] += 1
        native, duplicate = (second, first) if reverse else (first, second)
        native_before, duplicate_before = native.copy(), duplicate.copy()
        table.obsm["X_spatial"] = native
        table.obsm["spatial"] = duplicate
        expression = table.X

        if agree:
            shapes = read_shapes(table, layout.counts)
            assert table.obsm["spatial_pixels"] is native
            assert table.obsm["spatial"].dtype == np.float64
            assert not np.shares_memory(table.obsm["spatial"], native)
            assert not np.shares_memory(table.obsm["spatial"], duplicate)
            np.testing.assert_array_equal(table.obsm["spatial"], native_before.astype(np.float64) * 0.647)
            np.testing.assert_array_equal(shapes.geometry.x, native_before[:, 0])
            np.testing.assert_array_equal(shapes.geometry.y, native_before[:, 1])
        else:
            with pytest.raises(ReaderFormatError, match="disagreeing"):
                read_shapes(table, layout.counts)
            assert table.obsm["X_spatial"] is native
            assert table.obsm["spatial"] is duplicate
            assert "spatial_pixels" not in table.obsm
            assert "spatialdata_io_positions" not in table.uns

        assert table.X is expression
        np.testing.assert_array_equal(native, native_before)
        np.testing.assert_array_equal(duplicate, duplicate_before)

    @pytest.mark.parametrize("dtype", ["bool", "complex128", "object"])
    def test_invalid_optional_coordinate_dtype(self, tmp_path: Path, dtype: str) -> None:
        write_curio(tmp_path)
        layout = discover_layout(tmp_path)
        table = ad.read_h5ad(layout.counts)
        native = np.ones((3, 2), dtype=np.float64)
        duplicate = native.astype(dtype)
        table.obsm["X_spatial"] = native
        table.obsm["spatial"] = duplicate
        expression = table.X

        with pytest.raises(ReaderFormatError, match="spatial"):
            read_shapes(table, layout.counts)

        assert table.obsm["X_spatial"] is native
        assert table.obsm["spatial"] is duplicate
        assert table.X is expression
        assert "spatial_pixels" not in table.obsm
        assert "spatialdata_io_positions" not in table.uns
        np.testing.assert_array_equal(native, np.ones((3, 2)))
        np.testing.assert_array_equal(duplicate, np.ones((3, 2), dtype=dtype))

    @pytest.mark.parametrize("value", [-(2**60), -(2**53), 2**53, 2**60])
    @pytest.mark.parametrize("reverse", [False, True])
    def test_exact_large_integer_geometry(self, tmp_path: Path, value: int, *, reverse: bool) -> None:
        write_curio(tmp_path)
        layout = discover_layout(tmp_path)
        table = ad.read_h5ad(layout.counts)
        coordinates = np.zeros((3, 2), dtype=np.int64)
        coordinates[0, 0] = value
        floating = coordinates.astype(np.float64)
        native, duplicate = (floating, coordinates) if reverse else (coordinates, floating)
        table.obsm["X_spatial"] = native
        table.obsm["spatial"] = duplicate

        shapes = read_shapes(table, layout.counts)

        assert int(shapes.geometry.x.iloc[0]) == value
        assert table.obsm["spatial_pixels"] is native
