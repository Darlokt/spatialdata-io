"""Radius-one CODEX cell circles in the region pixel frame."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import geopandas as gpd
import pandas as pd
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

if TYPE_CHECKING:
    import anndata as ad
    import numpy as np
    import numpy.typing as npt


def _make_shapes(table: ad.AnnData) -> gpd.GeoDataFrame:
    """Create independent radius-one circles from normalized table coordinates."""
    # _read_cell_table establishes this owned, finite NumPy coordinate array.
    coordinates = cast("npt.NDArray[np.number]", table.obsm["spatial"])
    shapes = gpd.GeoDataFrame(
        {"geometry": gpd.points_from_xy(coordinates[:, 0], coordinates[:, 1]), "radius": 1},
        index=pd.Index(table.obs["cell_id"], name="cell_id"),
    )
    return ShapesModel.parse(shapes, transformations={"global": Identity()})
