"""Build one square polygon per observed DBiT grid location, in grid units.

Grid location ``(A, B)`` (one-based; A is x, B is y) occupies the cell
``[A - 1, A] x [B - 1, B]`` of the ``grid_pixels`` frame, one unit per
barcode-grid step. With ``include_border`` each square shrinks inward on every
side by ``border_scale / 8``, leaving a gap between neighbouring spots::

    A-1           A
      ┌────────────┐  B-1
      │  ┌──────┐  │
      │  │ spot │  │  inset = border_scale / 8 on every side
      │  └──────┘  │
      └────────────┘  B

Each ring starts at the corner nearest the origin and is clockwise with y
pointing up (``shapely.is_ccw`` is false). The polygons are indexed by
``pixel_id``. Vertices never depend on an image: the reader adds a scale from
``grid_pixels`` to ``image_pixels`` when an image is read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from spatialdata.models import ShapesModel

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt
    from spatialdata.transformations import BaseTransformation

_CORNERS = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64)
_INSET = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]], dtype=np.float64) * 0.125


def make_grid(
    coordinates: npt.NDArray[np.uint16],
    *,
    include_border: bool,
    border_scale: float,
    transformations: Mapping[str, BaseTransformation],
) -> gpd.GeoDataFrame:
    """Return grid-unit squares for ``(A, B)`` positions, indexed by their row number as ``pixel_id``."""
    # Broadcasting builds every closed ring at once, without Python polygons.
    vertices = coordinates[:, None, :] - 1 + _CORNERS
    if include_border:
        vertices += _INSET * border_scale

    # An inset below half an edge can still collapse after float rounding at
    # a distant grid position. Reject that loss of geometric information.
    if np.any(vertices[:, 2, :] <= vertices[:, 0, :]):
        message = "border_scale leaves no representable positive square width at these coordinates."
        raise ValueError(message)

    grid = gpd.GeoDataFrame(
        geometry=shapely.polygons(vertices),
        index=pd.Index(np.arange(len(coordinates), dtype=np.uint16), name="pixel_id"),
    )
    return ShapesModel.parse(grid, transformations=dict(transformations))
