"""Read Visium HD count matrices and link the final tables to their elements.

Bin and cell matrices are 10x HDF5 feature-barcode matrices read through the
shared 10x HDF5 reader, in source observation and feature order. A bin table
holds::

    X                  canonical float32 or float64 CSR sparse array of counts
    obs                in_tissue, array_row, array_col, pxl_row_in_fullres,
                       pxl_col_in_fullres (aligned positions), region, instance_id
    obs_names          s_<size:03d>um_<row:05d>_<col:05d>-1 barcodes
    var                gene_ids, feature_types, genome; var_names are gene symbols
    obsm["spatial"]    float64 nominal bin centres in global micrometres

``instance_id`` numbers the bins 1..n in table order; it is the shapes index
or the label value. Cell tables use the numeric cell ID of
``cellid_<id:09d>-1`` instead (``_cells``). Every table records its canonical
positions in ``uns["spatialdata_io_positions"]`` and is linked exactly once,
against the returned element's materialized index, after all filtering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from anndata import AnnData

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.counts import read_10x_h5_components
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.table import (
    TableLinkage,
    parse_linked_table,
    prepare_anndata_dataframe,
)
from spatialdata_io.readers.visium_hd._positions import read_positions
from spatialdata_io.readers.visium_hd._schemas import validate_probe_annotations
from spatialdata_io.readers.visium_hd._transforms import bin_centers

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.sparse import csr_array

    from spatialdata_io._positions import PositionsMetadata
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.visium_hd._layout import BinLayout
    from spatialdata_io.readers.visium_hd._transforms import SlideFrame


def read_counts(
    path: ArtifactFile, *, version: str | None, probe_counts: bool = False
) -> tuple[csr_array, pd.DataFrame, pd.DataFrame]:
    """Read shared sparse counts and validate the selected HD population."""
    context = ReaderErrorContext("visium_hd", "raw probe count matrix" if probe_counts else "count matrix", version)
    counts, obs, var = read_10x_h5_components(path, context=context)
    if not len(obs) or not len(var):
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"matrix dimensions {counts.shape}",
            expected="positive observation and feature dimensions",
        )
    # Strict bytewise order proves uniqueness on raw grids without a Python
    # string hash table. Unordered populations still get exact duplicate checks.
    names = obs.index
    if not (names[1:] > names[:-1]).all() and not names.is_unique:
        raise ReaderFormatError(context=context, path=path, found="duplicate barcodes", expected="unique barcodes")

    if probe_counts:
        validate_probe_annotations(obs, var, path, context=context)
        var["probe_name"] = var.index
        var.index = pd.Index(var["probe_ids"].array)

    return counts, obs, var


def read_bin_table(
    layout: BinLayout, shape: tuple[int, int], frame: SlideFrame, *, make_var_names_unique: bool, version: str | None
) -> AnnData:
    """Align sparse counts and nominal centres after source calibration is validated."""
    counts, obs, var = read_counts(layout.counts, version=version)
    positions = read_positions(layout.positions, shape, obs.index, layout.size_um, frame, version=version)
    return make_bin_table(counts, positions, var, layout.size_um, frame, make_var_names_unique=make_var_names_unique)


def make_bin_table(  # noqa: PLR0913 - Owned counts, aligned annotations, bin size and frame.
    counts: csr_array,
    positions: pd.DataFrame,
    var: pd.DataFrame,
    size_um: int,
    frame: SlideFrame,
    *,
    make_var_names_unique: bool,
) -> AnnData:
    """Build a bin count table after its positions have been aligned."""
    table = AnnData(counts, obs=positions, var=var)

    rows: NDArray[np.uint32] = positions["array_row"].to_numpy(dtype=np.uint32)
    cols: NDArray[np.uint32] = positions["array_col"].to_numpy(dtype=np.uint32)
    table.obsm["spatial"] = bin_centers(frame, rows, cols, float(size_um))

    if make_var_names_unique:
        table.var_names_make_unique()

    return table


def link_table(  # noqa: PLR0913, RUF100 - Explicit table, target, IDs, position meaning, source and version.
    table: AnnData,
    target: str,
    identifiers: pd.Index,
    *,
    position: str,
    counts_path: ArtifactFile,
    version: str | None,
) -> AnnData:
    """Link the final owned table once against its materialized target index."""
    context = ReaderErrorContext("visium_hd", "count matrix", version)
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
    table.obs["region"] = pd.Categorical([target] * table.n_obs, categories=pd.Index([target]))

    # Shared counts guarantee eager annotations; upstream types also include Dataset2D.
    prepare_anndata_dataframe(cast("pd.DataFrame", table.obs), context=context)
    prepare_anndata_dataframe(cast("pd.DataFrame", table.var), context=context)

    return parse_linked_table(
        table,
        TableLinkage((target,), "region", "instance_id"),
        context=context,
        expected_instance_ids={target: identifiers},
        path=counts_path,
    )
