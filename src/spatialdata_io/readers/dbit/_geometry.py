"""Construct the DBiT square grid in its image or unit-grid coordinate frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

if TYPE_CHECKING:
    import numpy.typing as npt


def _make_grid(
    coordinates: npt.NDArray[np.int64], *, scale: float, include_border: bool, border_scale: float
) -> gpd.GeoDataFrame:
    """Build clockwise squares from validated one-based A/x and B/y positions."""
    # Position one starts at the origin. Broadcasting keeps one closed ring
    # per observed spot without constructing individual Python polygons.
    corners = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64)
    vertices = (coordinates[:, None, :] - 1 + corners) * scale

    if include_border:
        # Preserve the original operation order and the 1/8-edge inset policy.
        border = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]], dtype=np.float64) * 0.125
        vertices += border * (scale * border_scale)

    # A valid inset below half an edge can still collapse after float rounding
    # at a distant grid position. Reject that loss of geometric information.
    if np.any(vertices[:, 2, :] <= vertices[:, 0, :]):
        message = "border_scale leaves no representable positive square width at these coordinates."
        raise ValueError(message)

    grid = gpd.GeoDataFrame(geometry=shapely.polygons(vertices), index=pd.RangeIndex(len(coordinates), name="pixel_id"))
    return ShapesModel.parse(grid, transformations={"global": Identity()})
