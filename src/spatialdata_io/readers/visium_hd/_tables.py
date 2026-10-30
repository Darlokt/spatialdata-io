"""Reader-owned count population checks and final Visium HD table linkage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd

from spatialdata_io.readers._utils.counts import read_10x_h5
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table, prepare_anndata_dataframe

if TYPE_CHECKING:
    from anndata import AnnData

    from spatialdata_io.readers._utils.paths import ArtifactFile


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


def link_table(table: AnnData, target: str, identifiers: pd.Index) -> AnnData:
    """Link the final owned table once against its materialized target index."""
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
