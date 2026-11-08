"""Finalize Stereo-seq tables: dropped cells, physical positions and linkage.

Every table reaches this module with its native DNB-grid coordinates in
``obsm["spatial"]``. :func:`finalize_positions` moves them to
``obsm["spatial_grid"]`` and stores float64 micrometres (grid value times the
pitch in nm divided by 1000) in ``obsm["spatial"]``, described in
``uns["spatialdata_io_positions"]``. :func:`link_table` adds categorical
``obs["region"]`` and ``obs["instance_id"]`` and links the table once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from spatialdata_io._positions import POSITIONS_KEY, FixedFrame, PositionsMetadata
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.table import (
    TableLinkage,
    finalize_count_arrays,
    parse_linked_table,
    prepare_anndata_dataframe,
)
from spatialdata_io.readers.stereoseq._schemas import EXPRESSION_LAYERS

if TYPE_CHECKING:
    from pathlib import Path

    import anndata as ad
    import numpy.typing as npt
    from scipy.sparse import csr_array

    from spatialdata_io.readers._utils.table import SparseDuplicateSummary


def select_cells(table: ad.AnnData, retained: pd.Index, *, context: ReaderErrorContext, path: Path) -> ad.AnnData:
    """Keep the rows of cells whose contours survived the geometry policy.

    Counts are re-finalized because dropping rows can make float32 exact again.
    """
    if len(retained) == table.n_obs:
        return table

    table = table[table.obs_names.isin(retained.astype(str))].copy()
    layers = ("exon",) if "exon" in table.layers else ()
    completed = finalize_count_arrays(
        (cast("csr_array", table.X), *(cast("csr_array", table.layers[name]) for name in layers)),
        context=context,
        path=path,
    )
    table.X = completed[0]
    table.layers.update(zip(layers, completed[1:], strict=True))
    return table


def finalize_positions(  # noqa: PLR0913 - position meaning and error context are separate inputs.
    table: ad.AnnData, pitch_nm: int, position: str, *, calibration: str, context: ReaderErrorContext, path: Path
) -> None:
    """Add micrometre positions to an owned table without copying expression."""
    if "spatial_grid" in table.obsm or POSITIONS_KEY in table.uns:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="reserved obsm['spatial_grid'] or uns['spatialdata_io_positions']",
            expected="available destinations for native coordinates and physical declarations",
        )

    # Producers validated exact 32-bit grid values and a pitch of at most 64
    # bits, so the float64 product is always finite.
    native = cast("npt.NDArray[np.integer | np.floating]", table.obsm["spatial"])
    physical = np.multiply(native, pitch_nm / 1000, dtype=np.float64)

    physical_description: FixedFrame = {
        "axes": ["x", "y"],
        "units": "micrometer",
        "frame": "global",
        "position": position,
        "calibration": {"global": calibration},
    }
    native_description: FixedFrame = {
        "axes": ["x", "y"],
        "units": "grid",
        "frame": "source_grid",
        "position": position,
    }
    metadata: PositionsMetadata = {
        "schema_version": 1,
        "coordinates": {"spatial": physical_description, "spatial_grid": native_description},
    }
    table.obsm["spatial_grid"] = table.obsm["spatial"]
    table.obsm["spatial"] = physical
    table.uns[POSITIONS_KEY] = metadata


def link_table(  # noqa: PLR0913 - linkage target, identity and error context are separate inputs.
    table: ad.AnnData,
    target: str,
    index: pd.Index,
    *,
    context: ReaderErrorContext,
    path: Path,
    duplicate_sums: list[SparseDuplicateSummary] | None = None,
) -> ad.AnnData:
    """Add the linkage columns once, then validate exact target membership."""
    obs = cast("pd.DataFrame", table.obs)
    instances = pd.Series(index.to_numpy(), index=obs.index)

    # Imported analysis may already contain linkage columns. Preserve matching
    # values, but never replace a conflicting annotation with an invented link.
    for column, expected in (("region", target), ("instance_id", instances)):
        if column in obs and not obs[column].eq(expected).fillna(value=False).all():
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"conflicting reserved column {column!r}",
                expected="absent or matching linkage metadata",
            )

    obs["region"] = pd.Categorical.from_codes(np.zeros(len(obs), dtype=np.int8), categories=pd.Index([target]))
    obs["instance_id"] = instances

    # Convert owned annotations only after their final mutation, immediately
    # before the shared AnnData/SpatialData compatibility and linkage boundary.
    prepare_anndata_dataframe(obs, context=context)
    prepare_anndata_dataframe(cast("pd.DataFrame", table.var), context=context)
    return parse_linked_table(
        table,
        TableLinkage((target,), "region", "instance_id"),
        context=context,
        path=path,
        expression_layers=tuple(name for name in EXPRESSION_LAYERS if name in table.layers),
        duplicate_sums=duplicate_sums,
        expected_instance_ids={target: index},
    )
