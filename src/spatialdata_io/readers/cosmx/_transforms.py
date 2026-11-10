"""Exact FOV placement, pixel-convention transforms and frame names for CosMx exports.

CosMx local coordinates are pixel indices with a reversed vertical axis: the
pixel at row ``r`` and column ``c`` of an ``H``-row FOV raster has local
coordinates ``(c, H - r)``. This was measured on the 2021 Lung5_Rep2 release
and on an AtoMx mouse-brain export, from label-pixel centroids against cell
metadata and from polygon extremes against label pixels. SpatialData places
pixel centres at ``(c + 0.5, r + 0.5)``, so the raster-to-local map is::

    [[1, 0, -0.5], [0, -1, H + 0.5], [0, 0, 1]]

Each FOV's local-to-global relation is a translation per axis, possibly with a
reversed axis: ``global = sign * local + offset``. Exports differ in the sign
of the y axis, so the sign and offset are derived from every metadata cell and
accepted only when one offset lies within all of their printed rounding
intervals. No affine is fitted and no tolerance is chosen. A pixel FOV
position (``fov_positions_file``) is the printed origin of a positive axis: it
must agree with the cell evidence, is preserved when feasible, and places FOVs
without cells.

Coordinate systems, all in pixels except ``global``::

    fov_<n>_pixels  FOV n's local pixel frame, for example "fov_2_pixels"
    global_pixels   the export's global pixel frame (without pixel_size_um)
    global          global micrometres: global pixels times pixel_size_um

``global`` always means micrometres, so an uncalibrated export has no
``global`` frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from spatialdata.transformations import Affine

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt
    import pandas as pd

    from spatialdata_io.readers._utils.paths import ArtifactFile

type Sign = Literal[1, -1]

GLOBAL = "global"
GLOBAL_PIXELS = "global_pixels"

_AXES = ("x", "y")
# Float64 decoding error of each correctly rounded value, in units of its magnitude.
_DECODING_ULPS = 2 * np.finfo(np.float64).eps


@dataclass(frozen=True, slots=True)
class AxisSamples:
    """Borrow one axis of owned cell coordinates, alongside its printed rounding bounds."""

    local: npt.NDArray[np.float64]
    global_: npt.NDArray[np.float64]
    local_half: npt.NDArray[np.float64]
    global_half: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FovPosition:
    """Hold one pixel FOV position and the rounding bound of its printed values."""

    x: float
    y: float
    x_half: float
    y_half: float


@dataclass(frozen=True, slots=True)
class Placement:
    """Map one FOV's local coordinates to global pixels by sign and offset per axis."""

    sign_x: Sign
    sign_y: Sign
    offset_x: float
    offset_y: float

    def local_to_global(self) -> Affine:
        """Return the exact local-to-global affine."""
        matrix = [[self.sign_x, 0, self.offset_x], [0, self.sign_y, self.offset_y], [0, 0, 1]]
        return Affine(np.asarray(matrix, dtype=np.float64), input_axes=_AXES, output_axes=_AXES)

    def global_to_local(self) -> Affine:
        """Return the exact inverse, used to show global points in a FOV frame."""
        matrix = [
            [self.sign_x, 0, -self.sign_x * self.offset_x],
            [0, self.sign_y, -self.sign_y * self.offset_y],
            [0, 0, 1],
        ]
        return Affine(np.asarray(matrix, dtype=np.float64), input_axes=_AXES, output_axes=_AXES)


def fov_frame(fov: int) -> str:
    """Return the name of one FOV's local pixel coordinate system."""
    return f"fov_{fov}_pixels"


def global_frame(pixel_size_um: float | None) -> str:
    """Return the global coordinate system: micrometres when calibrated, else pixels."""
    return GLOBAL_PIXELS if pixel_size_um is None else GLOBAL


def raster_to_local(height: int) -> Affine:
    """Map SpatialData pixel-center coordinates ``(c + 0.5, r + 0.5)`` to ``(c, H - r)``."""
    matrix = [[1, 0, -0.5], [0, -1, height + 0.5], [0, 0, 1]]
    return Affine(np.asarray(matrix, dtype=np.float64), input_axes=_AXES, output_axes=_AXES)


def _intervals(
    samples: AxisSamples, rows: npt.NDArray[np.bool_], path: ArtifactFile
) -> dict[Sign, tuple[float, float]]:
    """Return the offset interval consistent with every selected cell, for each sign."""
    local, global_ = samples.local[rows], samples.global_[rows]
    # A subnormal round-off allowance may underflow; printed bounds still apply.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        rounding_tolerance = (
            samples.local_half[rows]
            + samples.global_half[rows]
            + _DECODING_ULPS * np.maximum(np.abs(local), np.abs(global_))
        )
    if not np.isfinite(rounding_tolerance).all():
        raise _error(path, found="nonfinite printed-coordinate tolerance", expected="finite offset intervals")
    feasible: dict[Sign, tuple[float, float]] = {}
    for sign in (1, -1):
        with np.errstate(over="ignore", invalid="ignore"):
            offsets = global_ - sign * local
            lower, upper = offsets - rounding_tolerance, offsets + rounding_tolerance
        # An unrepresentable sign candidate cannot establish a finite placement.
        # The other sign may still have a valid, finite intersection.
        if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
            continue
        low, high = float(lower.max()), float(upper.min())
        if low <= high:
            feasible[sign] = (low, high)
    return feasible


def _error(path: ArtifactFile, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a placement contract violation with the metadata artifact."""
    return ReaderFormatError(
        context=ReaderErrorContext("cosmx", "FOV placement"), path=path, found=found, expected=expected
    )


def _dataset_sign(candidates: dict[int, dict[Sign, tuple[float, float]]], axis: str, path: ArtifactFile) -> Sign:
    """Require one sign shared by every FOV whose cells determine it."""
    determined = {next(iter(feasible)) for feasible in candidates.values() if len(feasible) == 1}
    if len(determined) != 1:
        raise _error(
            path,
            found=f"{axis} local-to-global signs {sorted(determined)}",
            expected="one sign determined by cells with distinct local coordinates",
        )
    return determined.pop()


def _midpoint(low: float, high: float) -> float:
    """Choose a midpoint without overflowing or halving same-sign subnormal endpoints."""
    return low + (high - low) / 2 if low >= 0 or high <= 0 else (low + high) / 2


def _position_offset(
    value: float, half: float, interval: tuple[float, float] | None, name: str, path: ArtifactFile
) -> float:
    """Intersect a finite printed origin interval with the optional cell evidence."""
    low, high = value - half, value + half
    if not np.isfinite([low, high]).all():
        raise _error(path, found=f"{name} nonfinite pixel-position interval", expected="finite offset intervals")
    if interval is None:
        return value
    low, high = max(interval[0], low), min(interval[1], high)
    if low > high:
        raise _error(
            path,
            found=f"{name} position {value} outside cell-derived {interval}",
            expected="FOV positions consistent with every cell",
        )
    # A rounded origin is an interval, not necessarily a feasible offset itself.
    return value if low <= value <= high else _midpoint(low, high)


def solve_placements(
    fovs: pd.Index,
    samples: tuple[AxisSamples, AxisSamples],
    *,
    required: frozenset[int],
    positions: Mapping[int, FovPosition],
    path: ArtifactFile,
) -> dict[int, Placement]:
    """Derive every required FOV's placement from all of its cells, cross-checked by positions."""
    candidates: tuple[dict[int, dict[Sign, tuple[float, float]]], dict[int, dict[Sign, tuple[float, float]]]] = ({}, {})
    for fov in fovs.unique().sort_values():
        # Pandas otherwise boxes a Python scalar as signed int64, overflowing
        # for valid large FOV IDs. Coordinates require one native boolean mask.
        rows = np.asarray(fovs == np.uint64(fov), dtype=np.bool_)
        for axis, axis_samples in enumerate(samples):
            feasible = _intervals(axis_samples, rows, path)
            if not feasible:
                raise _error(
                    path,
                    found=f"FOV {fov} {_AXES[axis]} coordinates inconsistent with any translation",
                    expected="finite global = ±local + offset within the printed precision of every cell",
                )
            candidates[axis][fov] = feasible

    signs = (_dataset_sign(candidates[0], "x", path), _dataset_sign(candidates[1], "y", path))
    placements: dict[int, Placement] = {}
    for fov in sorted(required | set(candidates[0])):
        position = positions.get(fov)
        offsets: list[float] = []
        for axis, sign in enumerate(signs):
            interval = candidates[axis].get(fov, {}).get(sign)
            if position is not None and sign == 1:
                value, half = (position.x, position.x_half) if axis == 0 else (position.y, position.y_half)
                offsets.append(_position_offset(value, half, interval, f"FOV {fov} {_AXES[axis]}", path))
            elif interval is not None:
                offsets.append(_midpoint(*interval))
            else:
                raise _error(
                    path,
                    found=f"FOV {fov} without cells or a pixel position",
                    expected="a placement for every FOV with rasters",
                )
        placements[fov] = Placement(signs[0], signs[1], offsets[0], offsets[1])
    return placements
