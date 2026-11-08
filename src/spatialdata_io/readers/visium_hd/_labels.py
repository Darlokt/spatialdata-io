"""Build exact bin-grid label rasters in global micrometres.

With ``bin_geometry="labels"`` each bin size becomes one ``uint32`` label image
with one pixel per bin: pixel ``(row, col)`` holds the table-order
``instance_id`` (1..n) of bin ``s_<size>um_<row>_<col>-1`` and 0 marks bins
absent from the selected matrix. The transform ``grid_um_to_global @ Scale(size)``
maps pixel corners to the nominal bin corners, so pixel centres are the bin
centres of ``obsm["spatial"]`` without a half-bin shift. The grid is eager
(4 bytes per bin: 45 MB for the 3350 x 3350 native grid), then chunked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from spatialdata.models import Labels2DModel
from spatialdata.transformations import Affine

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from xarray import DataArray

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers.visium_hd._transforms import SlideFrame


def make_bin_labels(  # noqa: PLR0913, RUF100 - Explicit grid, frame and raster inputs.
    frame: SlideFrame,
    rows: NDArray[np.uint32],
    cols: NDArray[np.uint32],
    shape: tuple[int, int],
    pitch: float,
    *,
    options: RasterOptions,
) -> DataArray:
    """Build labels with table-order IDs, zero background and exact bin corners.

    The eager allocation is one uint32 grid (45 MB for the 3350² native grid).
    Pixel centres and nominal square centres coincide without an added offset.
    """
    if len(rows) > np.iinfo(np.uint32).max:
        message = "The number of bin instances exceeds uint32 label capacity."
        raise ValueError(message)

    labels = np.zeros(shape, dtype=np.uint32)
    labels[rows, cols] = np.arange(1, len(rows) + 1, dtype=np.uint32)

    matrix = frame.grid_um_to_global @ np.diag([pitch, pitch, 1.0])
    # SpatialData annotates both single-scale and multiscale returns as a union.
    return cast(
        "DataArray",
        Labels2DModel.parse(
            labels,
            dims=("y", "x"),
            chunks=options.label_chunks(),
            transformations={"global": Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))},
        ),
    )
