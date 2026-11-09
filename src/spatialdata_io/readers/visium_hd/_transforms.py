"""Coordinate maps between the 2 µm barcode grid, image pixels and ``global`` micrometres.

Three coordinate spaces meet in a Visium HD output::

    spot (col, row)      2 µm barcode grid indices; spot (0, 0) is the first square
    image (x, y) px      full-resolution microscope pixels (``pxl_col/row_in_fullres``)
    global (x, y) µm     the returned physical frame

The slide is a manufactured continuous grid of 2 x 2 µm squares, so grid
micrometres are ``2 * (index + 0.5)`` at a square's centre. Space Ranger
registers the grid to the microscope image with one homography ``H``
(spot -> image pixels), which the reader refits from the 2 µm positions.

``global`` is grid micrometres, rigidly rotated (and, if the image is mirrored
relative to the slide, reflected) so that the tissue appears as in the
microscope image, with its origin at the fitted position of image pixel
``(0, 0)``. Distances and areas in ``global`` are therefore exact slide
micrometres. Image pixels reach ``global`` exactly through ``H^-1`` (vectors)
or approximately through the least-squares affine of ``SlideFrame`` (rasters,
whose residual is recorded). All maps act on homogeneous ``(x, y, 1)``
columns; SpatialData's pixel-corner convention is used throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from spatialdata_io.readers._utils.errors import ReaderFormatError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.errors import ReaderErrorContext

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


def apply_projective(
    matrix: NDArray[np.float64], xy: NDArray[np.float64], *, context: ReaderErrorContext
) -> NDArray[np.float64]:
    """Evaluate a validated homogeneous map on an N-by-2 coordinate array.

    ``context`` names the artifact whose coordinates are transformed.
    """
    denominator = xy @ matrix[2, :2] + matrix[2, 2]
    if np.any(denominator == 0) or not np.all(np.isfinite(denominator)):
        raise ReaderFormatError(
            context=context,
            found="zero or nonfinite projective denominator",
            expected="finite coordinates away from the projective horizon",
        )

    return (xy @ matrix[:2, :2].T + matrix[:2, 2]) / denominator[:, None]


def grid_to_global(frame: SlideFrame, xy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transform physical grid micrometres to image-oriented global micrometres."""
    return xy @ frame.grid_um_to_global[:2, :2].T + frame.grid_um_to_global[:2, 2]


def image_to_global(frame: SlideFrame, xy: NDArray[np.float64], *, context: ReaderErrorContext) -> NDArray[np.float64]:
    """Exactly reproject full-resolution image coordinates into global micrometres."""
    # Finite external vertices and centroids can still overflow during reprojection.
    with np.errstate(over="ignore", invalid="ignore"):
        spots = apply_projective(np.linalg.inv(frame.spot_to_image), xy, context=context)
        transformed = grid_to_global(frame, SPOT_PITCH_UM * (spots + 0.5))
    if not np.isfinite(transformed).all():
        raise ReaderFormatError(
            context=context,
            found="nonfinite transformed coordinates",
            expected="finite global coordinates representable in float64",
        )
    return transformed


def bin_centers(
    frame: SlideFrame,
    rows: NDArray[np.uint32],
    cols: NDArray[np.uint32],
    pitch: float,
) -> NDArray[np.float64]:
    """Return exact nominal bin centres, preserving the source table order."""
    grid = np.stack((cols, rows), axis=1, dtype=np.float64)
    return grid_to_global(frame, pitch * (grid + 0.5))


def fit_homography(
    source: NDArray[np.float64], target: NDArray[np.float64], *, context: ReaderErrorContext
) -> NDArray[np.float64]:
    """Fit a normalized projective map from well-spread validated correspondences."""
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_scale, target_scale = source.std(axis=0), target.std(axis=0)
    if np.any(source_scale == 0) or np.any(target_scale == 0):
        raise ReaderFormatError(
            context=context,
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
            context=context,
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
            context=context,
            found="singular fitted homography",
            expected="an invertible two-dimensional registration",
        )

    matrix = target_denormalizer @ normalized_matrix @ source_normalizer
    return matrix / matrix[2, 2]
