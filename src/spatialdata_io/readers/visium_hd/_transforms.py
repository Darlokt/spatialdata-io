"""Position-derived physical slide coordinates and registration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.paths import ArtifactFile

SPOT_PITCH_UM = 2.0
MAX_PROJECTIVE_ERROR_PX = 1e-9
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


def fit_slide_frame(positions: ArtifactFile, shape: tuple[int, int]) -> SlideFrame:
    """Fit the physical image frame in two bounded scans of all 2 µm positions.

    A centered covariance reduction fits the affine using every source bin.
    Nine well-spread correspondences fit the exact homography; the second scan
    validates it on every bin and measures the affine's registration residual.
    No complete position array or string index is retained.
    """
    # The positions storage owner also consumes bin_centers; defer this import.
    from spatialdata_io.readers.visium_hd._bins import iter_position_batches

    sample_rows = np.unique([0, shape[0] // 2, shape[0] - 1])
    sample_cols = np.unique([0, shape[1] // 2, shape[1] - 1])
    sample_source: list[NDArray[np.float64]] = []
    sample_target: list[NDArray[np.float64]] = []

    mean = np.zeros(4, dtype=np.float64)
    covariance = np.zeros((4, 4), dtype=np.float64)
    count = 0

    for batch in iter_position_batches(positions, shape):
        row = batch.column("array_row").to_numpy()
        col = batch.column("array_col").to_numpy()
        pixels = np.column_stack(
            (batch.column("pxl_col_in_fullres").to_numpy(), batch.column("pxl_row_in_fullres").to_numpy())
        )
        grid = np.column_stack((col, row)).astype(np.float64)

        selected = np.isin(row, sample_rows) & np.isin(col, sample_cols)
        sample_source.append(grid[selected])
        sample_target.append(pixels[selected])

        values = np.column_stack((pixels, SPOT_PITCH_UM * (grid + 0.5)))
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        delta = batch_mean - mean
        total = count + batch_count
        # Merge centered moments to avoid subtracting large, nearly equal raw sums.
        covariance += centered.T @ centered + np.outer(delta, delta) * (count * batch_count / total)
        mean += delta * (batch_count / total)
        count = total

    homography = fit_homography(np.concatenate(sample_source), np.concatenate(sample_target))
    linear = np.linalg.solve(covariance[:2, :2], covariance[:2, 2:]).T
    origin = mean[2:] - linear @ mean[:2]
    rigid, affine = _orient_frame(linear, origin)

    maximum = 0.0
    squared_sum = 0.0

    for batch in iter_position_batches(positions, shape):
        grid = np.column_stack((batch.column("array_col").to_numpy(), batch.column("array_row").to_numpy())).astype(
            np.float64
        )
        pixels = np.column_stack(
            (batch.column("pxl_col_in_fullres").to_numpy(), batch.column("pxl_row_in_fullres").to_numpy())
        )
        error_px = np.max(np.linalg.norm(apply_projective(homography, grid) - pixels, axis=1), initial=0.0)
        if error_px > MAX_PROJECTIVE_ERROR_PX:
            raise ReaderFormatError(
                context=ReaderErrorContext("visium_hd", "2 micrometre positions"),
                path=positions,
                found=f"projective registration residual {error_px:.6g} pixels",
                expected=f"one exact homography with residual <= {MAX_PROJECTIVE_ERROR_PX:g} pixels on every bin",
            )

        residual = pixels @ linear.T + origin - SPOT_PITCH_UM * (grid + 0.5)
        squared = np.einsum("ij,ij->i", residual, residual)
        maximum = max(maximum, float(np.sqrt(np.max(squared, initial=0.0))))
        squared_sum += float(squared.sum())

    for matrix in (homography, rigid, affine):
        matrix.flags.writeable = False
    return SlideFrame(homography, rigid, affine, maximum, float(np.sqrt(squared_sum / count)))


def _orient_frame(
    linear: NDArray[np.float64], origin: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Remove fitted rotation/reflection while keeping exact physical distances."""
    left, _, right = np.linalg.svd(linear)
    # Keep reflections: changing the determinant would mirror the tissue relative to its image.
    orthogonal = left @ right

    rigid = np.eye(3)
    rigid[:2, :2] = orthogonal.T
    rigid[:2, 2] = -orthogonal.T @ origin

    affine = np.eye(3)
    affine[:2, :2] = orthogonal.T @ linear
    return rigid, affine
