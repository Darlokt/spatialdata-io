"""Reader-owned count population checks and final Visium HD table linkage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from spatialdata_io._positions import POSITIONS_KEY, PositionsMetadata
from spatialdata_io.readers._utils.counts import read_10x_h5
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table, prepare_anndata_dataframe
from spatialdata_io.readers.visium_hd._positions import read_positions
from spatialdata_io.readers.visium_hd._transforms import bin_centers

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.visium_hd._layout import BinLayout
    from spatialdata_io.readers.visium_hd._transforms import SlideFrame


def read_counts(path: ArtifactFile) -> AnnData:
    """Read shared sparse counts and validate the selected HD population."""
    context = ReaderErrorContext("visium_hd", "count matrix")
    table = read_10x_h5(path, context=context, gex_only=False)
    if not table.n_obs or not table.n_vars:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"matrix dimensions {table.shape}",
            expected="positive observation and feature dimensions",
        )
    if not table.obs_names.is_unique:
        raise ReaderFormatError(context=context, path=path, found="duplicate barcodes", expected="unique barcodes")

    return table


def read_bin_table(
    layout: BinLayout, shape: tuple[int, int], frame: SlideFrame, *, make_var_names_unique: bool
) -> AnnData:
    """Align sparse counts and nominal centres after source calibration is validated."""
    table = read_counts(layout.counts)
    positions = read_positions(layout.positions, shape, table.obs_names, layout.size_um, frame)
    table.obs = positions

    rows: NDArray[np.uint32] = positions["array_row"].to_numpy(dtype=np.uint32)
    cols: NDArray[np.uint32] = positions["array_col"].to_numpy(dtype=np.uint32)
    table.obsm["spatial"] = bin_centers(frame, rows, cols, float(layout.size_um))

    if make_var_names_unique:
        table.var_names_make_unique()

    return table


def link_table(table: AnnData, target: str, identifiers: pd.Index, *, position: str) -> AnnData:
    """Link the final owned table once against its materialized target index."""
    if "spatial" in table.obsm:
        positions: PositionsMetadata = {
            "schema_version": 1,
            "coordinates": {
                "spatial": {
                    "axes": ["x", "y"],
                    "units": "micrometer",
                    "frame": "global",
                    "position": position,
                    "calibration": {"global": "Space Ranger manufactured bin grid and validated slide fit"},
                }
            },
        }
        table.uns[POSITIONS_KEY] = positions
    context = ReaderErrorContext("visium_hd", target)
    table.obs["region"] = pd.Categorical([target] * table.n_obs, categories=pd.Index([target]))

    # Shared counts guarantee eager annotations; upstream types also include Dataset2D.
    prepare_anndata_dataframe(cast("pd.DataFrame", table.obs), context=context)
    prepare_anndata_dataframe(cast("pd.DataFrame", table.var), context=context)

    return parse_linked_table(
        table,
        TableLinkage((target,), "region", "instance_id"),
        context=context,
        expected_instance_ids={target: identifiers},
    )
