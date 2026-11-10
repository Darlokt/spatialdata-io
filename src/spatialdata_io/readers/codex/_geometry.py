"""Build the CODEX cell shapes: radius-one circles at the native cell positions.

The cell table has positions but no segmentation boundaries, so each cell is a
circle of radius one native pixel centred at its ``x, y``, indexed by its
uint64 cell ID. The circles are display geometry, not measured cells; their
coordinates stay native and the transformation maps them to the output frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
from spatialdata.models import ShapesModel

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from spatialdata.transformations import BaseTransformation


def make_shapes(
    cell_ids: npt.NDArray[np.uint64],
    coordinates: npt.NDArray[np.number],
    *,
    frame: str,
    transformation: BaseTransformation,
) -> gpd.GeoDataFrame:
    """Create radius-one circles indexed by cell ID from native ``x, y`` positions."""
    shapes = gpd.GeoDataFrame(
        {"geometry": gpd.points_from_xy(coordinates[:, 0], coordinates[:, 1]), "radius": 1},
        index=pd.Index(cell_ids, name="cell_id"),
    )
    return ShapesModel.parse(shapes, transformations={frame: transformation})
