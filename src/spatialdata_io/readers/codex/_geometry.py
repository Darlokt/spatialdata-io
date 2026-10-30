"""Radius-one CODEX cell circles in the region pixel frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
from spatialdata.models import ShapesModel

if TYPE_CHECKING:
    import anndata as ad
    import numpy as np
    import numpy.typing as npt
    from spatialdata.transformations import BaseTransformation


def _make_shapes(
    table: ad.AnnData, coordinates: npt.NDArray[np.number], *, frame: str, transformation: BaseTransformation
) -> gpd.GeoDataFrame:
    """Create independent radius-one circles from normalized table coordinates."""
    shapes = gpd.GeoDataFrame(
        {"geometry": gpd.points_from_xy(coordinates[:, 0], coordinates[:, 1]), "radius": 1},
        index=pd.Index(table.obs["cell_id"].to_numpy(), name="cell_id"),
    )
    return ShapesModel.parse(shapes, transformations={frame: transformation})
