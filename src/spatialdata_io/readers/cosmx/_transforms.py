"""Exact FOV placement and pixel-convention transforms for CosMx exports.

CosMx local coordinates are pixel indices with a reversed vertical axis: the
pixel at row ``r`` and column ``c`` of an ``H``-row FOV raster has local
coordinates ``(c, H - r)``. This was measured on the 2021 Lung5_Rep2 release
and on an AtoMx mouse-brain export, from label-pixel centroids against cell
metadata and from polygon extremes against label pixels.

Each FOV's local-to-global relation is a translation per axis, possibly with a
reversed axis: ``global = sign * local + offset``. Exports differ in the sign
of the y axis, so the sign and offset are derived from every metadata cell and
accepted only when one offset lies within all of their printed rounding
intervals. No affine is fitted and no tolerance is chosen.
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

    from spatialdata_io.readers._utils.paths import ArtifactFile

type Sign = Literal[1, -1]

_AXES = ("x", "y")
# Float64 decoding error of each correctly rounded value, in units of its magnitude.
_DECODING_ULPS = 2 * np.finfo(np.float64).eps


@dataclass(frozen=True, slots=True)
class AxisSamples:
    """Own one axis of every metadata cell: values and their printed rounding bounds."""

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


def raster_to_local(height: int) -> Affine:
    """Map SpatialData pixel-center coordinates ``(c + 0.5, r + 0.5)`` to ``(c, H - r)``."""
    matrix = [[1, 0, -0.5], [0, -1, height + 0.5], [0, 0, 1]]
    return Affine(np.asarray(matrix, dtype=np.float64), input_axes=_AXES, output_axes=_AXES)


def _intervals(samples: AxisSamples, rows: npt.NDArray[np.bool_]) -> dict[Sign, tuple[float, float]]:
    """Return the offset interval consistent with every selected cell, for each sign."""
    local, global_ = samples.local[rows], samples.global_[rows]
    budget = (
        samples.local_half[rows]
        + samples.global_half[rows]
        + _DECODING_ULPS * np.maximum(np.abs(local), np.abs(global_))
    )
    feasible: dict[Sign, tuple[float, float]] = {}
    for sign in (1, -1):
        offsets = global_ - sign * local
        low, high = float((offsets - budget).max()), float((offsets + budget).min())
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


def solve_placements(
    fovs: npt.NDArray[np.int64],
    samples: tuple[AxisSamples, AxisSamples],
    *,
    required: frozenset[int],
    positions: Mapping[int, FovPosition],
    path: ArtifactFile,
) -> dict[int, Placement]:
    """Derive every required FOV's placement from all of its cells, cross-checked by positions."""
    candidates: list[dict[int, dict[Sign, tuple[float, float]]]] = [{}, {}]
    for fov in np.unique(fovs).tolist():
        rows = fovs == fov
        for axis, axis_samples in enumerate(samples):
            feasible = _intervals(axis_samples, rows)
            if not feasible:
                raise _error(
                    path,
                    found=f"FOV {fov} {_AXES[axis]} coordinates inconsistent with any translation",
                    expected="global = ±local + offset within the printed precision of every cell",
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
                # Evidenced exports with a positive axis print the FOV origin as the exact offset.
                value, half = (position.x, position.x_half) if axis == 0 else (position.y, position.y_half)
                if interval is not None and not interval[0] - half <= value <= interval[1] + half:
                    raise _error(
                        path,
                        found=f"FOV {fov} {_AXES[axis]} position {value} outside cell-derived {interval}",
                        expected="FOV positions consistent with every cell",
                    )
                offsets.append(value)
            elif interval is not None:
                offsets.append((interval[0] + interval[1]) / 2)
            else:
                raise _error(
                    path,
                    found=f"FOV {fov} without cells or a pixel position",
                    expected="a placement for every FOV with rasters",
                )
        placements[fov] = Placement(signs[0], signs[1], offsets[0], offsets[1])
    return placements
