"""Position-derived physical slide coordinates and registration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from numpy.typing import NDArray

SPOT_PITCH_UM = 2.0
_PROJECTIVE_COEFFICIENTS = 8


@dataclass(frozen=True, slots=True)
class SlideFrame:
    """Owned slide registration; matrices map homogeneous x/y coordinates.

    The global origin is the fitted image affine's pixel-zero origin. The exact
    inverse homography can differ there by the measured projective residual.
    """

    spot_to_image: NDArray[np.float64]
    grid_um_to_global: NDArray[np.float64]
    image_pixels_to_global: NDArray[np.float64]
    max_residual_um: float
    rms_residual_um: float


def apply_projective(matrix: NDArray[np.float64], xy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate a validated homogeneous map on an N-by-2 coordinate array."""
    denominator = xy @ matrix[2, :2] + matrix[2, 2]
    if np.any(denominator == 0) or not np.all(np.isfinite(denominator)):
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "coordinate transformation"),
            found="zero or nonfinite projective denominator",
            expected="finite coordinates away from the projective horizon",
        )

    return (xy @ matrix[:2, :2].T + matrix[:2, 2]) / denominator[:, None]


def grid_to_global(frame: SlideFrame, xy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transform physical grid micrometres to image-oriented global micrometres."""
    return apply_projective(frame.grid_um_to_global, xy)


def image_to_global(frame: SlideFrame, xy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exactly reproject full-resolution image coordinates into global micrometres."""
    spots = apply_projective(np.linalg.inv(frame.spot_to_image), xy)
    return grid_to_global(frame, SPOT_PITCH_UM * (spots + 0.5))


def bin_centers(
    frame: SlideFrame,
    rows: NDArray[np.uint32],
    cols: NDArray[np.uint32],
    pitch: float,
) -> NDArray[np.float64]:
    """Return exact nominal bin centres, preserving the source table order."""
    grid = np.column_stack((cols, rows)).astype(np.float64)
    return grid_to_global(frame, pitch * (grid + 0.5))


def fit_homography(source: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fit a normalized projective map from well-spread validated correspondences."""
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_scale, target_scale = source.std(axis=0), target.std(axis=0)
    if np.any(source_scale == 0) or np.any(target_scale == 0):
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "2 micrometre positions"),
            found="degenerate coordinate extent",
            expected="a two-dimensional grid and invertible registration",
        )

    normalized_source = (source - source_mean) / source_scale
    normalized_target = (target - target_mean) / target_scale
    # Alternating rows encode the x and y equations with the final coefficient fixed to one.
    design = np.zeros((2 * len(source), 8), dtype=np.float64)
    homogeneous = np.column_stack((normalized_source, np.ones(len(source))))
    design[::2, :3] = homogeneous
    design[1::2, 3:6] = homogeneous
    design[::2, 6:] = -normalized_target[:, 0, None] * normalized_source
    design[1::2, 6:] = -normalized_target[:, 1, None] * normalized_source

    solution, _, rank, _ = np.linalg.lstsq(design, normalized_target.ravel(), rcond=None)
    if rank != _PROJECTIVE_COEFFICIENTS:
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "2 micrometre positions"),
            found=f"projective fit rank {rank}",
            expected="eight independent projective coefficients",
        )

    normalized_matrix = np.append(solution, 1).reshape(3, 3)
    source_normalizer = np.eye(3)
    source_normalizer[:2, :2] = np.diag(1 / source_scale)
    source_normalizer[:2, 2] = -source_mean / source_scale
    target_denormalizer = np.eye(3)
    target_denormalizer[:2, :2] = np.diag(target_scale)
    target_denormalizer[:2, 2] = target_mean
    if np.linalg.matrix_rank(normalized_matrix) != normalized_matrix.shape[0]:
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "2 micrometre positions"),
            found="singular fitted homography",
            expected="an invertible two-dimensional registration",
        )

    matrix = target_denormalizer @ normalized_matrix @ source_normalizer
    return matrix / matrix[2, 2]
