"""Construct Curio bead geometry without changing coordinate meaning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
from spatialdata.models import ShapesModel

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    import anndata as ad

    from spatialdata_io.readers._utils.paths import ArtifactFile


def _read_shapes(table: ad.AnnData, path: ArtifactFile) -> gpd.GeoDataFrame:
    """Validate duplicate embeddings and construct exact radius-10 bead geometry."""
    xy = table.obsm.get("spatial")
    duplicate = table.obsm.get("X_spatial")
    if (
        not isinstance(xy, np.ndarray)
        or xy.shape != (table.n_obs, 2)
        or xy.dtype.kind not in "iuf"
        or not np.isfinite(xy).all()
        or not isinstance(duplicate, np.ndarray)
        or not np.array_equal(xy, duplicate)
    ):
        raise ReaderFormatError(
            context=ReaderErrorContext("curio", "spatial coordinates"),
            path=path,
            found="missing, invalid, or disagreeing spatial arrays",
            expected="identical finite numeric spatial and X_spatial arrays of shape (n_obs, 2), ordered (x, y)",
        )

    # Shapely stores float64 coordinates. Reject a lossy conversion before geometry construction.
    if xy.dtype.kind in "iu":
        candidates = xy[(xy > 2**53) | (xy < -(2**53))]
        loss = any(int(value) != int(float(value)) for value in candidates)
    elif xy.dtype.itemsize > np.dtype(np.float64).itemsize:
        with np.errstate(over="ignore"):
            loss = not np.array_equal(xy, xy.astype(np.float64))
    else:
        loss = False

    if loss:
        raise ReaderFormatError(
            context=ReaderErrorContext("curio", "spatial coordinates"),
            path=path,
            found="spatial coordinates lose precision in float64 geometry",
            expected="coordinates losslessly representable as float64",
        )

    # The embeddings agree, so retain one coordinate representation in the owned table.
    del table.obsm["X_spatial"]
    shapes = gpd.GeoDataFrame(
        {"geometry": gpd.points_from_xy(xy[:, 0], xy[:, 1]), "radius": 10.0},
        index=table.obs_names.copy(),
    )
    return ShapesModel.parse(shapes)
