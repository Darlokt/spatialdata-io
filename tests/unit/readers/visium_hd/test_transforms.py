"""Physical coordinate and position-derived registration contracts."""

from __future__ import annotations

import numpy as np
import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers.visium_hd._transforms import (
    SlideFrame,
    apply_projective,
    bin_centers,
    fit_homography,
    grid_to_global,
    image_to_global,
)
from tests.integration.readers.visium_hd._fixtures import HOMOGRAPHY

_CONTEXT = ReaderErrorContext("visium_hd", "test coordinates", "4.1.0")


class TestApplyProjective:
    """Homogeneous maps preserve coordinate order and reject the horizon."""

    def test_exact_perspective(self) -> None:
        xy = np.array([[0.0, 0.0], [12, 3], [100, 200]])
        expected = (HOMOGRAPHY @ np.column_stack((xy, np.ones(3))).T).T
        np.testing.assert_array_equal(
            apply_projective(HOMOGRAPHY, xy, context=_CONTEXT), expected[:, :2] / expected[:, 2:]
        )

    @pytest.mark.parametrize("denominator", [0.0, np.nan, np.inf])
    def test_horizon(self, denominator: float) -> None:
        matrix = np.eye(3)
        matrix[2, 2] = denominator
        with pytest.raises(ReaderFormatError, match="projective denominator"):
            apply_projective(matrix, np.zeros((2, 2)), context=_CONTEXT)


class TestFitHomography:
    """Normalization retains accuracy over full-sized reflected capture grids."""

    @pytest.mark.parametrize("extent", [17, 3350, 5524])
    def test_normalized_fit(self, extent: int) -> None:
        rows, cols = np.meshgrid([0, extent / 2, extent - 1], [0, extent / 2, extent - 1])
        source = np.column_stack((cols.ravel(), rows.ravel()))
        target = apply_projective(HOMOGRAPHY, source, context=_CONTEXT)
        fitted = fit_homography(source, target, context=_CONTEXT)
        check = np.random.default_rng(0).uniform(0, extent - 1, (1000, 2))
        np.testing.assert_allclose(
            apply_projective(fitted, check, context=_CONTEXT),
            apply_projective(HOMOGRAPHY, check, context=_CONTEXT),
            rtol=0,
            atol=1e-9,
        )

    def test_degenerate(self) -> None:
        with pytest.raises(ReaderFormatError, match="degenerate"):
            fit_homography(np.zeros((9, 2)), np.zeros((9, 2)), context=_CONTEXT)

    def test_rank_deficient(self) -> None:
        points = np.column_stack((np.arange(9, dtype=float), np.arange(9, dtype=float)))
        with pytest.raises(ReaderFormatError, match="rank"):
            fit_homography(points, points, context=_CONTEXT)


class TestGridToGlobal:
    """The rigid frame preserves physical lengths and area under reflection."""

    def test_reflection_rotation(self) -> None:
        matrix = np.array([[0.0, 1, 3], [1, 0, -5], [0, 0, 1]])
        frame = SlideFrame(np.eye(3), matrix, np.eye(3), 0, 0)
        np.testing.assert_array_equal(grid_to_global(frame, np.array([[2.0, 7]])), [[10, -3]])

    def test_affine_matches_homogeneous_map(self) -> None:
        angle = 0.37
        matrix = np.array(
            [[np.cos(angle), np.sin(angle), 1.25], [np.sin(angle), -np.cos(angle), -8.75], [0.0, 0.0, 1.0]]
        )
        frame = SlideFrame(np.eye(3), matrix, np.eye(3), 0, 0)
        xy = np.random.default_rng(7).uniform(-10_000, 10_000, (100, 2))
        np.testing.assert_array_equal(grid_to_global(frame, xy), apply_projective(matrix, xy, context=_CONTEXT))


class TestImageToGlobal:
    """Inverse-H reprojection adds one micrometre to grid-index coordinates."""

    def test_exact_corner_and_center(self) -> None:
        frame = SlideFrame(HOMOGRAPHY, np.eye(3), np.eye(3), 0, 0)
        spots = np.array([[-0.5, -0.5], [0, 0], [16.5, 16.5]])
        np.testing.assert_allclose(
            image_to_global(frame, apply_projective(HOMOGRAPHY, spots, context=_CONTEXT), context=_CONTEXT),
            [[0, 0], [1, 1], [34, 34]],
            rtol=0,
            atol=1e-13,
        )

    def test_finite_source_coordinate_overflow(self) -> None:
        frame = SlideFrame(np.diag([0.5, 0.5, 1.0]), np.eye(3), np.eye(3), 0, 0)
        with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ReaderFormatError):
            image_to_global(frame, np.array([[1e308, 1e308]]), context=_CONTEXT)


class TestBinCenters:
    """Grid indices give physical centres even at nominal coarse edges."""

    def test_source_order_and_nominal_center(self) -> None:
        frame = SlideFrame(np.eye(3), np.eye(3), np.eye(3), 0, 0)
        rows, cols = np.array([2, 0, 1], np.uint32), np.array([2, 1, 0], np.uint32)
        np.testing.assert_array_equal(bin_centers(frame, rows, cols, 16), [[40, 40], [24, 8], [8, 24]])

    def test_uint32_indices_above_signed_range(self) -> None:
        matrix = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, -5.0], [0.0, 0.0, 1.0]])
        frame = SlideFrame(np.eye(3), matrix, np.eye(3), 0, 0)
        rows = np.array([2**31 + 1, 0], dtype=np.uint32)
        cols = np.array([0, 2**31 + 2], dtype=np.uint32)
        np.testing.assert_array_equal(bin_centers(frame, rows, cols, 2), [[2**32 + 6, -4], [4, 2**32]])
