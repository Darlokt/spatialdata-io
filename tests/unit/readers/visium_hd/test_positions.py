"""Bounded positions, registration and exact source-order grid alignment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium_hd._positions import (
    fit_slide_frame,
    iter_position_batches,
    read_grid_shape,
    read_positions,
)
from spatialdata_io.readers.visium_hd._transforms import SlideFrame, apply_projective
from tests.integration.readers.visium_hd._fixtures import GRID_SHAPE, HOMOGRAPHY, write_bundle

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _positions(root: Path) -> ArtifactFile:
    return ArtifactFile(root / "binned_outputs/square_002um/spatial/tissue_positions.parquet")


def _frame() -> SlideFrame:
    return SlideFrame(HOMOGRAPHY, np.eye(3), np.eye(3), 0.0, 0.0)


class TestReadGridShape:
    """Parquet footer grid bounds and bounded fallback."""

    def test_footer_statistics(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        mocker.patch.object(pq.ParquetFile, "iter_batches", side_effect=AssertionError("unexpected positions decoding"))
        assert read_grid_shape(_positions(tmp_path)) == (17, 17)

    def test_projected_fallback(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        path = _positions(tmp_path)
        pq.write_table(pq.read_table(path), path, write_statistics=False)
        assert read_grid_shape(path) == (17, 17)

    @pytest.mark.parametrize("invalid", ["empty", "null"])
    def test_invalid(self, tmp_path: Path, invalid: str) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        path = _positions(tmp_path)
        table = pq.read_table(path)
        if invalid == "empty":
            table = table.slice(0, 0)
        else:
            table = table.set_column(
                2, pa.field("array_row", pa.uint32()), pa.array([None] * table.num_rows, type=pa.uint32())
            )
        pq.write_table(table, path, write_statistics=False)
        with pytest.raises(ReaderFormatError):
            read_grid_shape(path)

    def test_malformed_parquet(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "positions.parquet")
        path.write_bytes(b"not parquet")
        with pytest.raises(ReaderFormatError, match="positions Parquet"):
            read_grid_shape(path)


class TestIterPositionBatches:
    """Complete grid coverage, projection and resource ownership."""

    def test_projected_reordered(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        path = _positions(tmp_path)
        table = pq.read_table(path).take(np.arange(288, -1, -1)).append_column("extra", pa.array([True] * 289))
        pq.write_table(table, path)
        batches = list(iter_position_batches(path, (17, 17)))
        assert sum(batch.num_rows for batch in batches) == 289
        assert "extra" not in batches[0].schema.names

    @pytest.mark.parametrize("mode", ["missing", "row", "col", "duplicate", "nonfinite"])
    def test_invalid_grid(self, tmp_path: Path, mode: str) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        path = _positions(tmp_path)
        table = pq.read_table(path)
        if mode == "missing":
            table = table.slice(1)
        elif mode == "nonfinite":
            column = np.zeros(289)
            column[0] = np.nan
            table = table.set_column(4, table.schema.field(4), pa.array(column))
        elif mode == "duplicate":
            table = table.take(pa.array([1, *range(1, 289)]))
        else:
            name = "array_row" if mode == "row" else "array_col"
            column = table[name].to_numpy().copy()
            column[0] = 17
            table = table.set_column(table.schema.get_field_index(name), table.schema.field(name), pa.array(column))
        pq.write_table(table, path)
        with pytest.raises(ReaderFormatError):
            list(iter_position_batches(path, (17, 17)))

    def test_projected_batches_close_on_grid_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        path = _positions(tmp_path)
        table = pq.read_table(path)
        # Duplicate lies in a later decoded batch, exercising cross-batch coverage.
        indices = list(range(289))
        indices[40] = 0
        pq.write_table(table.take(pa.array(indices)), path)
        source = pq.ParquetFile(path)
        mocker.patch("spatialdata_io.readers.visium_hd._positions.POSITION_BATCH_SIZE", 37)
        mocker.patch("spatialdata_io.readers.visium_hd._positions.pq.ParquetFile", return_value=source)
        mocker.patch(
            "spatialdata_io.readers.visium_hd._positions.pq.read_table",
            side_effect=AssertionError("unbounded positions read"),
        )
        with pytest.raises(ReaderFormatError, match="duplicate grid"):
            list(iter_position_batches(path, (17, 17)))
        assert source.closed

    def test_malformed_parquet(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "positions.parquet")
        path.write_bytes(b"not parquet")
        with pytest.raises(ReaderFormatError, match="positions Parquet"):
            list(iter_position_batches(path, (17, 17)))


class TestReadPositions:
    """Exact barcode alignment in matrix observation order."""

    def test_exact_source_order_and_empty(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        path = _positions(tmp_path)
        names = pd.Index(["s_002um_00001_00003-1", "s_002um_00000_00000-1"])
        result = read_positions(path, (17, 17), names, 2, _frame())
        assert result.index.equals(names)
        assert_array_equal(result["array_row"], [1, 0])
        assert_array_equal(result["array_col"], [3, 0])
        assert str(result["pxl_col_in_fullres"].dtype) == "double[pyarrow]"
        assert read_positions(path, (17, 17), pd.Index([], dtype=str), 2, _frame()).empty

    @pytest.mark.parametrize("size", [2, 6, 8, 16])
    def test_producer_registration_and_partial_edges(self, tmp_path: Path, size: int) -> None:
        write_bundle(tmp_path, sizes=(2, 6, 8, 16), segmented=False)
        path = ArtifactFile(tmp_path / f"binned_outputs/square_{size:03d}um/spatial/tissue_positions.parquet")
        source = pq.read_table(path)
        names = pd.Index(source["barcode"].to_pylist())
        factor = size // 2
        extent = (17 + factor - 1) // factor
        result = read_positions(path, (extent, extent), names[::-1], size, _frame())
        assert result.index.equals(names[::-1])
        assert_array_equal(result["pxl_col_in_fullres"], source["pxl_col_in_fullres"].to_numpy()[::-1])
        assert_array_equal(result["pxl_row_in_fullres"], source["pxl_row_in_fullres"].to_numpy()[::-1])
        # Even the partial far corner uses the whole bin's origin offset.
        corners = apply_projective(HOMOGRAPHY, np.array([[0.0, 0.0], [factor - 1.0, factor - 1.0]]))
        last = apply_projective(HOMOGRAPHY, np.array([[factor * (extent - 1.0)] * 2]))[0]
        assert_allclose(
            result.iloc[0][["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float64),
            last + (corners[1] - corners[0]) / 2,
            rtol=0,
            atol=1e-12,
        )

    @pytest.mark.parametrize("shift", [0.5e-9, 2e-9, 1_000_000.0])
    def test_existing_registration_tolerance(self, tmp_path: Path, shift: float) -> None:
        write_bundle(tmp_path, segmented=False)
        path = ArtifactFile(tmp_path / "binned_outputs/square_008um/spatial/tissue_positions.parquet")
        source = pq.read_table(path)
        values = source["pxl_col_in_fullres"].to_numpy().copy()
        values += shift
        pq.write_table(source.set_column(5, source.schema.field(5), pa.array(values)), path)
        names = pd.Index(["s_008um_00000_00000-1"])
        if shift < 1e-9:
            assert len(read_positions(path, (5, 5), names, 8, _frame())) == 1
        else:
            with pytest.raises(ReaderFormatError, match=r"1e-09.*cross-bin registration residual"):
                read_positions(path, (5, 5), names, 8, _frame())

    @pytest.mark.parametrize("mode", ["mixed", "clipped-edge", "projected-centre"])
    def test_wrong_registration_in_unselected_bins(self, tmp_path: Path, mode: str) -> None:
        write_bundle(tmp_path, segmented=False)
        path = ArtifactFile(tmp_path / "binned_outputs/square_008um/spatial/tissue_positions.parquet")
        source = pq.read_table(path)
        values = source["pxl_col_in_fullres"].to_numpy().copy()
        if mode == "mixed":
            values[-1] += 1.0
        else:
            # The last partial bin contains only spot (16, 16). Neither its
            # captured centre nor the projected nominal centre is the vendor offset.
            point = 16.0 if mode == "clipped-edge" else 17.5
            values[-1] = apply_projective(HOMOGRAPHY, np.array([[point, point]]))[0, 0]
        pq.write_table(source.set_column(5, source.schema.field(5), pa.array(values)), path)
        with pytest.raises(ReaderFormatError, match="cross-bin registration"):
            read_positions(path, (5, 5), pd.Index(["s_008um_00000_00000-1"]), 8, _frame())

    def test_registration_failure_closes_source(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_bundle(tmp_path, segmented=False)
        path = ArtifactFile(tmp_path / "binned_outputs/square_008um/spatial/tissue_positions.parquet")
        source = pq.ParquetFile(path)
        mocker.patch("spatialdata_io.readers.visium_hd._positions.pq.ParquetFile", return_value=source)
        frame = SlideFrame(np.eye(3), np.eye(3), np.eye(3), 0, 0)
        with pytest.raises(ReaderFormatError, match="cross-bin registration"):
            read_positions(path, (5, 5), pd.Index(["s_008um_00000_00000-1"]), 8, frame)
        assert source.closed

    def test_selected_barcode_mismatch(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        path = _positions(tmp_path)
        table = pq.read_table(path)
        names = table["barcode"].to_pylist()
        names[0] = "wrong"
        pq.write_table(table.set_column(0, table.schema.field(0), pa.array(names)), path)
        with pytest.raises(ReaderFormatError, match="barcode differs"):
            read_positions(path, (17, 17), pd.Index(["s_002um_00000_00000-1"]), 2, _frame())


class TestFitSlideFrame:
    """Every position contributes to the affine and validates the projective map."""

    def test_all_positions_reference(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, segmented=False)
        path = ArtifactFile(tmp_path / "binned_outputs/square_002um/spatial/tissue_positions.parquet")
        frame = fit_slide_frame(path, GRID_SHAPE)
        source = pq.read_table(path)
        grid = np.column_stack((source["array_col"].to_numpy(), source["array_row"].to_numpy())).astype(float)
        pixels = np.column_stack((source["pxl_col_in_fullres"].to_numpy(), source["pxl_row_in_fullres"].to_numpy()))
        reference = np.linalg.lstsq(np.column_stack((pixels, np.ones(len(pixels)))), 2 * (grid + 0.5), rcond=None)[0].T
        left, _, right = np.linalg.svd(reference[:, :2])
        rotation = left @ right
        np.testing.assert_allclose(frame.spot_to_image, HOMOGRAPHY, rtol=0, atol=1e-12)
        np.testing.assert_allclose(frame.grid_um_to_global[:2, :2], rotation.T, rtol=0, atol=1e-14)
        np.testing.assert_allclose(frame.grid_um_to_global[:2, 2], -rotation.T @ reference[:, 2], rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            frame.image_pixels_to_global[:2, :2], rotation.T @ reference[:, :2], rtol=0, atol=1e-14
        )
        np.testing.assert_array_equal(frame.image_pixels_to_global[:2, 2], 0)
        residual = np.column_stack((pixels, np.ones(len(pixels)))) @ reference.T - 2 * (grid + 0.5)
        errors = np.linalg.norm(residual, axis=1)
        assert frame.max_residual_um == pytest.approx(errors.max(), abs=1e-12)
        assert frame.rms_residual_um == pytest.approx(np.sqrt(np.mean(errors**2)), abs=1e-12)
        assert np.linalg.det(frame.grid_um_to_global[:2, :2]) < 0
        for matrix in (frame.spot_to_image, frame.grid_um_to_global, frame.image_pixels_to_global):
            assert not matrix.flags.writeable

    def test_unsampled_position_is_validated(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, segmented=False)
        path = ArtifactFile(tmp_path / "binned_outputs/square_002um/spatial/tissue_positions.parquet")
        table = pq.read_table(path)
        values = table["pxl_col_in_fullres"].to_numpy().copy()
        values[18] += 1e-6
        table = table.set_column(table.column_names.index("pxl_col_in_fullres"), "pxl_col_in_fullres", pa.array(values))
        pq.write_table(table, path)
        with pytest.raises(ReaderFormatError, match=r"residual.*1e-09"):
            fit_slide_frame(path, GRID_SHAPE)

    def test_singular_affine_registration(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, segmented=False)
        path = ArtifactFile(tmp_path / "binned_outputs/square_002um/spatial/tissue_positions.parquet")
        table = pq.read_table(path)
        # Collinear image positions cannot define an image-to-slide frame.
        index = table.column_names.index("pxl_row_in_fullres")
        table = table.set_column(index, "pxl_row_in_fullres", table["pxl_col_in_fullres"])
        pq.write_table(table, path)
        with pytest.raises(ReaderFormatError, match=r"rank|singular"):
            fit_slide_frame(path, GRID_SHAPE)
