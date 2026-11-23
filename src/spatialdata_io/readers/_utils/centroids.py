"""Physical XY coordinates for zero-based regionprops pixel-index centroids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from spatialdata_io.readers._utils.errors import ReaderFormatError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.paths import ArtifactFile


def scale_regionprops_centroids(
    physical: NDArray[np.float64],
    spacing_xy_um: tuple[float, float],
    *,
    frame: str,
    path: ArtifactFile,
    context: ReaderErrorContext,
) -> None:
    """Shift pixel indices to pixel centres and scale each axis in float64.

    The reader owns this writable float64 copy of the source coordinates and
    supplies validated spacing. Only the selected rows are modified.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        physical += 0.5
        nonzero = physical != 0
        physical *= spacing_xy_um
    if not np.isfinite(physical).all():
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"nonfinite physical positions in {frame!r}",
            expected="finite float64 physical XY",
        )
    if np.any((physical == 0) & nonzero):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="nonzero shifted pixel coordinates becoming zero physical positions",
            expected="nonzero float64 physical XY for nonzero shifted pixel coordinates",
        )
