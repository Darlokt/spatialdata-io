"""Exact CosMx pixel convention and FOV placement from printed-precision intervals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.cosmx._transforms import (
    AxisSamples,
    FovPosition,
    Placement,
    Sign,
    _midpoint,
    fov_frame,
    global_frame,
    raster_to_local,
    solve_placements,
)

if TYPE_CHECKING:
    from pathlib import Path

_AXES = ("x", "y")


def _apply(matrix: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    homogeneous = np.column_stack([np.asarray(points, dtype=np.float64), np.ones(len(points))])
    return (matrix @ homogeneous.T).T[:, :2]


def _samples(local: list[float], global_: list[float], half: float = 5e-7) -> AxisSamples:
    count = len(local)
    return AxisSamples(
        np.asarray(local, dtype=np.float64),
        np.asarray(global_, dtype=np.float64),
        np.full(count, half),
        np.full(count, half),
    )


def _solve(
    tmp_path: Path,
    fovs: list[int],
    samples: tuple[AxisSamples, AxisSamples],
    *,
    required: frozenset[int] = frozenset(),
    positions: dict[int, FovPosition] | None = None,
) -> dict[int, Placement]:
    return solve_placements(
        np.asarray(fovs, dtype=np.int64),
        samples,
        required=required,
        positions=positions or {},
        path=ArtifactFile(tmp_path / "metadata.csv"),
    )


class TestRasterToLocal:
    """Pixel centers map to column index and reversed row index."""

    def test_pixel_centers(self) -> None:
        matrix = raster_to_local(6).to_affine_matrix(input_axes=_AXES, output_axes=_AXES)
        # SpatialData centers pixel (row r, column c) at (c + 0.5, r + 0.5).
        np.testing.assert_array_equal(_apply(matrix, [(2.5, 1.5), (6.5, 4.5), (0.5, 5.5)]), [[2, 5], [6, 2], [0, 1]])


class TestPlacement:
    """Forward and inverse maps are exact inverses for either sign."""

    @pytest.mark.parametrize(("sign_x", "sign_y"), [(1, 1), (1, -1), (-1, 1)])
    def test_inverse(self, sign_x: Sign, sign_y: Sign) -> None:
        placement = Placement(sign_x, sign_y, 1000.25, 5000.5)
        forward = placement.local_to_global().to_affine_matrix(input_axes=_AXES, output_axes=_AXES)
        backward = placement.global_to_local().to_affine_matrix(input_axes=_AXES, output_axes=_AXES)
        np.testing.assert_array_equal(backward @ forward, np.eye(3))
        np.testing.assert_array_equal(_apply(forward, [(2, 3)]), [[1000.25 + sign_x * 2, 5000.5 + sign_y * 3]])


class TestMidpoint:
    """Finite interval endpoints retain a feasible midpoint across the float64 range."""

    @pytest.mark.parametrize(
        ("low", "high", "expected"),
        [
            (5e-324, 5e-324, 5e-324),
            (-5e-324, -5e-324, -5e-324),
            (-5e-324, 1.5e-323, 5e-324),
            (-1.5e308, 1.5e308, 0),
            (1.5e308, 1.5e308, 1.5e308),
            (-1.5e308, -1.5e308, -1.5e308),
        ],
    )
    def test_finite_range(self, low: float, high: float, expected: float) -> None:
        assert _midpoint(low, high) == expected


class TestSolvePlacements:
    """Signs and offsets are exact consequences of every cell's rounding interval."""

    def test_positive_translation_per_fov(self, tmp_path: Path) -> None:
        placements = _solve(
            tmp_path,
            [2, 2, 10, 10],
            (
                _samples([2, 6, 2, 6], [1002, 1006, 3002.25, 3006.25]),
                _samples([5, 2, 5, 2], [5005.5, 5002.5, 7005, 7002]),
            ),
        )
        assert placements == {2: Placement(1, 1, 1000.0, 5000.5), 10: Placement(1, 1, 3000.25, 7000.0)}

    def test_reversed_y_axis(self, tmp_path: Path) -> None:
        placements = _solve(tmp_path, [1, 1], (_samples([2, 6], [102, 106]), _samples([5, 2], [995, 998])))
        assert placements == {1: Placement(1, -1, 100.0, 1000.0)}

    def test_integer_rounded_local_values_narrow_the_offset(self, tmp_path: Path) -> None:
        # True offset 100.3: locals are rounded integers (±0.5), globals are exact to 1e-7.
        local = [1, 2, 3]
        global_ = [100.3 + 1.2, 100.3 + 1.8, 100.3 + 3.4]
        x = AxisSamples(np.asarray(local, float), np.asarray(global_), np.full(3, 0.5), np.full(3, 5e-8))
        placements = _solve(tmp_path, [1, 1, 1], (x, _samples([1, 2, 3], [11, 12, 13])))
        assert placements[1].offset_x == pytest.approx(100.3, abs=0.2)

    def test_inconsistent_cells_fail_without_tolerance(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="inconsistent with any translation"):
            _solve(tmp_path, [1, 1], (_samples([2, 6], [102, 106.00001]), _samples([5, 2], [15, 12])))

    def test_single_cell_fov_takes_the_dataset_sign(self, tmp_path: Path) -> None:
        placements = _solve(
            tmp_path, [1, 1, 2], (_samples([2, 6, 4], [102, 106, 204]), _samples([5, 2, 3], [15, 12, 23]))
        )
        assert placements[2] == Placement(1, 1, 200.0, 20.0)

    @pytest.mark.parametrize(
        ("fovs", "x", "y", "match"),
        [
            ([1], _samples([2], [102]), _samples([5], [15]), "x local-to-global signs"),
            ([1, 1, 2, 2], _samples([2, 6, 2, 6], [102, 106, 102, 106]), _samples([1, 2, 1, 2], [11, 12, 9, 8]), "y"),
        ],
    )
    def test_undetermined_or_mixed_signs_fail(
        self, tmp_path: Path, fovs: list[int], x: AxisSamples, y: AxisSamples, match: str
    ) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            _solve(tmp_path, fovs, (x, y))

    def test_feasible_positions_are_preserved_and_place_fovs_without_cells(self, tmp_path: Path) -> None:
        positions = {1: FovPosition(100.0, 10.0, 5e-7, 5e-7), 3: FovPosition(300.5, 30.25, 5e-7, 5e-7)}
        placements = _solve(
            tmp_path,
            [1, 1],
            (_samples([2, 6], [102, 106]), _samples([5, 2], [15, 12])),
            required=frozenset({3}),
            positions=positions,
        )
        assert placements == {1: Placement(1, 1, 100.0, 10.0), 3: Placement(1, 1, 300.5, 30.25)}

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize(
        ("position", "half", "expected"),
        [(99.6, 0.35, 99.925), (100.4, 0.35, 100.075), (100.025, 0.001, 100.025)],
    )
    def test_position_and_cell_intersection(
        self, tmp_path: Path, axis: int, position: float, half: float, expected: float
    ) -> None:
        samples = (_samples([2, 6], [102, 106], half=0.05), _samples([2, 6], [102, 106], half=0.05))
        origin = FovPosition(position, 100, half, 0) if axis == 0 else FovPosition(100, position, 0, half)
        placement = _solve(tmp_path, [1, 1], samples, positions={1: origin})[1]
        offset = (placement.offset_x, placement.offset_y)[axis]
        assert offset == pytest.approx(expected, rel=0, abs=1e-12)
        assert position - half <= offset <= position + half
        # Check every original cell, independently of the solver's reduced interval.
        source = samples[axis]
        allowance = source.local_half + source.global_half
        allowance += 2 * np.finfo(np.float64).eps * np.maximum(np.abs(source.local), np.abs(source.global_))
        assert np.all(np.abs(source.global_ - source.local - offset) <= allowance)

    def test_position_interval_touching_cell_interval(self, tmp_path: Path) -> None:
        samples = (_samples([0, 4], [0, 4], half=0), _samples([0, 4], [0, 4], half=0))
        placement = _solve(tmp_path, [1, 1], samples, positions={1: FovPosition(1, -1, 1, 1)})[1]
        assert placement == Placement(1, 1, 0, 0)

    def test_large_finite_offset_midpoint_does_not_overflow(self, tmp_path: Path) -> None:
        samples = _samples([0, 1e300], [1.5e308, 1.50000001e308], half=0)
        with np.errstate(over="raise"):
            placement = _solve(tmp_path, [1, 1], (samples, samples))[1]
        assert np.isfinite([placement.offset_x, placement.offset_y]).all()
        assert placement.offset_x == pytest.approx(1.5e308, rel=1e-15)

    def test_position_must_agree_with_cells(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="outside cell-derived"):
            _solve(
                tmp_path,
                [1, 1],
                (_samples([2, 6], [102, 106]), _samples([5, 2], [15, 12])),
                positions={1: FovPosition(100.5, 10.0, 5e-7, 5e-7)},
            )

    def test_unplaceable_required_fov_fails(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="FOV 3 without cells or a pixel position"):
            _solve(
                tmp_path, [1, 1], (_samples([2, 6], [102, 106]), _samples([5, 2], [15, 12])), required=frozenset({3})
            )

    def test_reversed_axis_positions_are_not_used_as_offsets(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="FOV 3 without cells"):
            _solve(
                tmp_path,
                [1, 1],
                (_samples([2, 6], [102, 106]), _samples([5, 2], [995, 998])),
                required=frozenset({3}),
                positions={3: FovPosition(300.0, 30.0, 0.5, 0.5)},
            )

    def test_unrepresentable_wrong_sign_does_not_hide_finite_placement(self, tmp_path: Path) -> None:
        samples = _samples([1e308, 1.1e308], [1e308, 1.1e308])
        with np.errstate(all="raise"):
            placement = _solve(tmp_path, [1, 1], (samples, samples))[1]
        assert placement == Placement(1, 1, 0, 0)
        np.testing.assert_array_equal(
            placement.local_to_global().to_affine_matrix(input_axes=_AXES, output_axes=_AXES), np.eye(3)
        )

    def test_unrepresentable_offset_is_not_a_feasible_sign(self, tmp_path: Path) -> None:
        samples = _samples([-1e308, -9e307], [1e308, 1.1e308])
        with np.errstate(all="raise"), pytest.raises(ReaderFormatError, match="inconsistent with any translation"):
            _solve(tmp_path, [1, 1], (samples, samples))

    def test_overflowing_interval_endpoints_fail(self, tmp_path: Path) -> None:
        samples = _samples([0, 1], [1.7e308, 1.7e308], half=5e307)
        with np.errstate(all="raise"), pytest.raises(ReaderFormatError, match="finite global"):
            _solve(tmp_path, [1, 1], (samples, samples))

    def test_nonfinite_tolerance_fails(self, tmp_path: Path) -> None:
        samples = _samples([0, 1], [10, 11], half=1e308)
        with np.errstate(all="raise"), pytest.raises(ReaderFormatError, match="nonfinite printed-coordinate tolerance"):
            _solve(tmp_path, [1, 1], (samples, samples))

    @pytest.mark.parametrize("with_cells", [False, True])
    def test_nonfinite_position_interval_fails(self, tmp_path: Path, *, with_cells: bool) -> None:
        samples = _samples([2, 6], [102, 106])
        fov = 1 if with_cells else 3
        with pytest.raises(ReaderFormatError, match="nonfinite pixel-position interval"):
            _solve(
                tmp_path,
                [1, 1],
                (samples, samples),
                required=frozenset({fov}),
                positions={fov: FovPosition(1.7e308, 100, 5e307, 0.5)},
            )


class TestFovFrame:
    """Each FOV's local pixel frame is named descriptively by its number."""

    def test_name(self) -> None:
        assert fov_frame(2) == "fov_2_pixels"
        assert fov_frame(10) == "fov_10_pixels"


class TestGlobalFrame:
    """Global is micrometres; without a scale the global frame is a pixel frame."""

    def test_names(self) -> None:
        assert global_frame(None) == "global_pixels"
        assert global_frame(0.18) == "global"
