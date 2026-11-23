"""Build Seeker bead circles in native pixels with a transform to global micrometres.

Takara's Seeker FAQ gives a nominal 10 µm bead diameter
(https://www.takarabio.com/learning-centers/spatial-omics/seeker-resources/seeker-faqs).
Shapely stores the circle centres as float64. The bead radius in pixels uses
the same calibration as the physical positions in the table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
from spatialdata.models import ShapesModel
from spatialdata.transformations import Scale

from spatialdata_io.readers.curio._positions import MICROMETRES_PER_PIXEL

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray

BEAD_DIAMETER_MICROMETRES = 10.0
BEAD_RADIUS_PIXELS = BEAD_DIAMETER_MICROMETRES / 2 / MICROMETRES_PER_PIXEL


def make_shapes(xy: NDArray[np.integer | np.floating], bead_ids: pd.Index[str]) -> gpd.GeoDataFrame:
    """Build bead circles from validated native positions and borrow their identifiers."""
    shapes = gpd.GeoDataFrame(
        {"geometry": gpd.points_from_xy(xy[:, 0], xy[:, 1]), "radius": BEAD_RADIUS_PIXELS},
        # A separate Index object: ShapesModel keeps it, and its mutable name must not alias obs_names.
        index=bead_ids.copy(),
    )
    scale = Scale([MICROMETRES_PER_PIXEL, MICROMETRES_PER_PIXEL], axes=("x", "y"))
    return ShapesModel.parse(shapes, transformations={"global": scale})
