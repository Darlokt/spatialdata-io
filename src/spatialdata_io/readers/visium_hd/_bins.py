"""Sparse bin matrices and nominal physical grid centres."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from spatialdata_io.readers.visium_hd._metadata import read_bin_scale_factors
from spatialdata_io.readers.visium_hd._positions import read_positions
from spatialdata_io.readers.visium_hd._tables import read_counts
from spatialdata_io.readers.visium_hd._transforms import bin_centers

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray

    from spatialdata_io.readers.visium_hd._layout import BinLayout
    from spatialdata_io.readers.visium_hd._transforms import SlideFrame


def read_bin_table(
    layout: BinLayout, shape: tuple[int, int], frame: SlideFrame, *, make_var_names_unique: bool
) -> AnnData:
    """Read one selected raw/filtered matrix and its physical bin centres."""
    read_bin_scale_factors(layout.scale_factors, layout.size_um)
    table = read_counts(layout.counts)
    positions = read_positions(layout.positions, shape, table.obs_names, layout.size_um, frame)
    table.obs = positions

    rows: NDArray[np.uint32] = positions["array_row"].to_numpy(dtype=np.uint32)
    cols: NDArray[np.uint32] = positions["array_col"].to_numpy(dtype=np.uint32)
    table.obsm["spatial"] = bin_centers(frame, rows, cols, float(layout.size_um))

    if make_var_names_unique:
        table.var_names_make_unique()

    return table
