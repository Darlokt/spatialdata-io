"""Physical coordinate and position-derived registration contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium_hd._transforms import (
    SlideFrame,
    apply_projective,
    bin_centers,
    fit_homography,
    fit_slide_frame,
    grid_to_global,
    image_to_global,
)
from tests.integration.readers.visium_hd._fixtures import GRID_SHAPE, HOMOGRAPHY, write_bundle

if TYPE_CHECKING:
    from pathlib import Path


class TestApplyProjective:
    """Homogeneous maps preserve coordinate order and reject the horizon."""

    def test_exact_perspective(self) -> None:
        xy = np.array([[0.0, 0.0], [12, 3], [100, 200]])
        expected = (HOMOGRAPHY @ np.column_stack((xy, np.ones(3))).T).T
        np.testing.assert_array_equal(apply_projective(HOMOGRAPHY, xy), expected[:, :2] / expected[:, 2:])

    @pytest.mark.parametrize("denominator", [0.0, np.nan, np.inf])
    def test_horizon(self, denominator: float) -> None:
        matrix = np.eye(3)
        matrix[2, 2] = denominator
        with pytest.raises(ReaderFormatError, match="projective denominator"):
            apply_projective(matrix, np.zeros((2, 2)))


class TestFitHomography:
    """Normalization retains accuracy over full-sized reflected capture grids."""

    @pytest.mark.parametrize("extent", [17, 3350, 5524])
    def test_normalized_fit(self, extent: int) -> None:
        rows, cols = np.meshgrid([0, extent / 2, extent - 1], [0, extent / 2, extent - 1])
        source = np.column_stack((cols.ravel(), rows.ravel()))
        target = apply_projective(HOMOGRAPHY, source)
        fitted = fit_homography(source, target)
        check = np.random.default_rng(0).uniform(0, extent - 1, (1000, 2))
        np.testing.assert_allclose(
            apply_projective(fitted, check), apply_projective(HOMOGRAPHY, check), rtol=0, atol=1e-9
        )

    def test_degenerate(self) -> None:
        with pytest.raises(ReaderFormatError, match="degenerate"):
            fit_homography(np.zeros((9, 2)), np.zeros((9, 2)))

    def test_rank_deficient(self) -> None:
        points = np.column_stack((np.arange(9, dtype=float), np.arange(9, dtype=float)))
        with pytest.raises(ReaderFormatError, match="rank"):
            fit_homography(points, points)


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


class TestGridToGlobal:
    """The rigid frame preserves physical lengths and area under reflection."""

    def test_reflection_rotation(self) -> None:
        matrix = np.array([[0.0, 1, 3], [1, 0, -5], [0, 0, 1]])
        frame = SlideFrame(np.eye(3), matrix, np.eye(3), 0, 0)
        np.testing.assert_array_equal(grid_to_global(frame, np.array([[2.0, 7]])), [[10, -3]])


class TestImageToGlobal:
    """Inverse-H reprojection adds one micrometre to grid-index coordinates."""

    def test_exact_corner_and_center(self) -> None:
        frame = SlideFrame(HOMOGRAPHY, np.eye(3), np.eye(3), 0, 0)
        spots = np.array([[-0.5, -0.5], [0, 0], [16.5, 16.5]])
        np.testing.assert_allclose(
            image_to_global(frame, apply_projective(HOMOGRAPHY, spots)), [[0, 0], [1, 1], [34, 34]], rtol=0, atol=1e-13
        )


class TestBinCenters:
    """Grid indices give physical centres even at nominal coarse edges."""

    def test_source_order_and_nominal_center(self) -> None:
        frame = SlideFrame(np.eye(3), np.eye(3), np.eye(3), 0, 0)
        rows, cols = np.array([2, 0, 1], np.uint32), np.array([2, 1, 0], np.uint32)
        np.testing.assert_array_equal(bin_centers(frame, rows, cols, 16), [[40, 40], [24, 8], [8, 24]])
