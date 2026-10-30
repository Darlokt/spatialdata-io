"""Curio bead coordinate validation and exact geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.curio._geometry import _read_shapes
from spatialdata_io.readers.curio._layout import _discover_layout
from tests.integration.readers.curio._fixtures import write_curio

if TYPE_CHECKING:
    from pathlib import Path


class TestReadShapes:
    """Reject invalid coordinates before constructing a spatial element."""

    @pytest.mark.parametrize("kind", ["missing_duplicate", "shape", "dtype", "nonfinite", "disagree"])
    def test_invalid_spatial_arrays(self, tmp_path: Path, kind: str) -> None:
        write_curio(tmp_path)
        layout = _discover_layout(tmp_path)
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
            _read_shapes(table, layout.counts)

    @pytest.mark.parametrize("destination", ["spatial_pixels", "spatialdata_io_positions", "X_SPATIAL"])
    def test_reserved_destinations_and_unsupported_profile(self, tmp_path: Path, destination: str) -> None:
        write_curio(tmp_path)
        layout = _discover_layout(tmp_path)
        table = ad.read_h5ad(layout.counts)
        if destination == "spatialdata_io_positions":
            table.uns[destination] = {"source": "retained"}
        else:
            table.obsm[destination] = np.ones((3, 2))
        with pytest.raises(ReaderFormatError, match=r"destination|profile"):
            _read_shapes(table, layout.counts)

    def test_native_buffer_and_expression_ownership(self, tmp_path: Path) -> None:
        write_curio(tmp_path, duplicate_coordinates=False)
        layout = _discover_layout(tmp_path)
        table = ad.read_h5ad(layout.counts)
        native = table.obsm["X_spatial"]
        expression = table.X
        _read_shapes(table, layout.counts)
        assert table.obsm["spatial_pixels"] is native
        assert table.X is expression
        np.testing.assert_array_equal(native, [[30, 4], [10, 8], [20, 2]])
        assert table.obsm["spatial"].dtype == np.float64
        assert table.obsm["spatial"] is not native

    @pytest.mark.parametrize("value", [2**53 + 1, -(2**53 + 1), 2**64 - 1])
    def test_integer_coordinate_precision_loss(self, tmp_path: Path, value: int) -> None:
        write_curio(tmp_path)
        layout = _discover_layout(tmp_path)
        table = ad.read_h5ad(layout.counts)
        coordinates = np.zeros((3, 2), dtype=np.uint64 if value >= 2**63 else np.int64)
        coordinates[0, 0] = value
        table.obsm["spatial"] = coordinates
        table.obsm["X_spatial"] = coordinates.copy()

        with pytest.raises(ReaderFormatError, match="lose precision"):
            _read_shapes(table, layout.counts)

    def test_exact_large_integer_geometry(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        layout = _discover_layout(tmp_path)
        table = ad.read_h5ad(layout.counts)
        coordinates = np.zeros((3, 2), dtype=np.int64)
        coordinates[0, 0] = 2**60
        table.obsm["spatial"] = coordinates
        table.obsm["X_spatial"] = coordinates.copy()

        shapes = _read_shapes(table, layout.counts)

        assert int(shapes.geometry.x.iloc[0]) == 2**60

    @pytest.mark.parametrize("exact", [True, False])
    def test_extended_float_geometry(self, tmp_path: Path, *, exact: bool) -> None:
        if np.finfo(np.longdouble).eps == np.finfo(np.float64).eps:
            pytest.skip("This platform has no floating type wider than float64.")
        write_curio(tmp_path)
        layout = _discover_layout(tmp_path)
        table = ad.read_h5ad(layout.counts)
        coordinates = np.ones((3, 2), dtype=np.longdouble)
        if not exact:
            coordinates[0, 0] += np.finfo(np.longdouble).eps
        table.obsm["spatial"] = coordinates
        table.obsm["X_spatial"] = coordinates.copy()

        if exact:
            assert _read_shapes(table, layout.counts).geometry.x.tolist() == [1.0] * 3
        else:
            with pytest.raises(ReaderFormatError, match="lose precision"):
                _read_shapes(table, layout.counts)
