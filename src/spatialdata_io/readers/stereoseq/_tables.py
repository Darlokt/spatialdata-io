"""Finalize owned Stereo-seq positions and link completed populations once."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from spatialdata_io._positions import POSITIONS_KEY, FixedFrame, PositionsMetadata
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table, prepare_anndata_dataframe

if TYPE_CHECKING:
    import anndata as ad
    import numpy.typing as npt


def finalize_positions(table: ad.AnnData, pitch_nm: int, position: str, *, calibration: str) -> None:
    """Finalize an owned, source-validated table without copying expression.

    Coordinates and pitch come exclusively from the existing source parsers.
    Native coordinates stay available for raw/analysis matching and targets.
    """
    if "spatial_grid" in table.obsm or POSITIONS_KEY in table.uns:
        raise ReaderFormatError(
            context=ReaderErrorContext("stereoseq", "position annotations"),
            found="reserved spatial_grid or spatialdata_io_positions annotation",
            expected="available destinations for native coordinates and physical declarations",
        )

    # Every producer validates an eager real native embedding. Convert directly
    # from that owned array, including after a required population slice.
    native = cast("npt.NDArray[np.integer | np.floating]", table.obsm["spatial"])
    physical = np.multiply(native, pitch_nm / 1000, dtype=np.float64)
    if not np.isfinite(physical).all():
        raise ReaderFormatError(
            context=ReaderErrorContext("stereoseq", "physical positions"),
            found="nonfinite transformed coordinates",
            expected="finite float64 micrometre positions",
        )

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


def link_table(table: ad.AnnData, target: str, index: pd.Index) -> ad.AnnData:
    """Complete owned annotations once, then validate exact target membership."""
    context = ReaderErrorContext("stereoseq", target)
    obs = cast("pd.DataFrame", table.obs)
    instances = pd.Series(index.to_numpy(), index=obs.index)

    # Imported analysis may already contain linkage columns. Preserve matching
    # values, but never replace a conflicting annotation with an invented link.
    for column, expected in (("region", target), ("instance_id", instances)):
        if column in obs and not obs[column].eq(expected).fillna(value=False).all():
            raise ReaderFormatError(
                context=context,
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
        expression_layers=tuple(name for name in ("exon", "counts", "log1p") if name in table.layers),
        expected_instance_ids={target: index},
    )
