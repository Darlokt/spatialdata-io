"""Build seqFISH cell boundary polygons and centroid circles.

``<ROI>_Boundaries.geojson`` stores one GeoJSON feature per cell, identified by
its top-level ``id``, with vertices in DAPI pixel coordinates (see
:mod:`._schemas`). Vertices are kept as stored; the DAPI pixel size registers
them in the ROI's micrometre frame as a ``Scale`` transformation. Polygons are
reordered to the table's cell order, and the feature IDs must equal the table's
cell IDs exactly, under every invalid-geometry policy.

Circles are derived from ``<ROI>_CellCoordinates.csv``: the centroid
``(center_x, center_y)`` in µm and the radius ``sqrt(area / pi)`` of a disc with
the cell's area in µm², so they need no calibration and use the identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.geometry import geojson_polygons, invalid_geometry_mask, raise_invalid_geometries
from spatialdata_io.readers._utils.metadata import read_json_object
from spatialdata_io.readers.seqfish._schemas import BOUNDARIES_CONTEXT, CELLS_CONTEXT, boundary_features

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata.transformations import Scale

    from spatialdata_io.readers._utils.geometry import InvalidGeometryPolicy
    from spatialdata_io.readers._utils.paths import ArtifactFile


def read_boundaries(
    path: ArtifactFile, *, instance_ids: pd.Index, frame: str, scale: Scale, policy: InvalidGeometryPolicy
) -> gpd.GeoDataFrame:
    """Read eager GeoJSON polygons and align their independently encoded IDs."""
    document = read_json_object(path, context=BOUNDARIES_CONTEXT)
    features, identifiers = boundary_features(document, path)
    geometries = geojson_polygons([feature["geometry"] for feature in features], context=BOUNDARIES_CONTEXT, path=path)
    # Keep arbitrary annotation IDs exact, including nullable integers, rather
    # than allowing pandas to infer a lossy floating column from JSON values.
    shapes = gpd.GeoDataFrame(
        [
            {"geometry": geometry, **(feature["properties"] or {})}
            for geometry, feature in zip(geometries, features, strict=True)
        ],
        dtype=object,
    )
    shapes.index = identifiers.rename("instance_id")
    if len(shapes) != len(instance_ids) or (shapes.index.get_indexer(instance_ids) < 0).any():
        raise ReaderFormatError(
            context=BOUNDARIES_CONTEXT,
            path=path,
            found="boundary/cell ID disagreement",
            expected="exactly the table cell IDs",
        )
    if not shapes.index.equals(instance_ids):
        shapes = shapes.loc[instance_ids]

    if policy != "keep":
        invalid = invalid_geometry_mask(shapes.geometry.to_numpy())
        if policy == "raise":
            raise_invalid_geometries(invalid, shapes.index.tolist(), context=BOUNDARIES_CONTEXT, path=path)
        elif invalid.any():
            shapes = shapes.loc[~invalid]
            if shapes.empty:
                raise ReaderFormatError(
                    context=BOUNDARIES_CONTEXT,
                    path=path,
                    found="all boundaries dropped, leaving an empty element",
                    expected="at least one retained cell boundary",
                )
    return ShapesModel.parse(shapes, transformations={frame: scale})


def make_circles(table: AnnData, *, frame: str, cells: ArtifactFile) -> gpd.GeoDataFrame:
    """Use physical centroids and square-micrometre areas for cell circles."""
    # The table parser owns eager, finite, float64 XY coordinates and areas.
    coordinates = cast("np.ndarray", table.obsm["spatial"])
    # The reader-created table is eager; AnnData's backed Dataset2D alternative
    # does not apply at this private producer boundary.
    obs = cast("pd.DataFrame", table.obs)
    # ShapesModel validates radii with NumPy, which Arrow-backed values do not support.
    areas = obs["area"].to_numpy(dtype=np.float64)
    if (areas <= 0).any():
        raise ReaderFormatError(
            context=CELLS_CONTEXT,
            path=cells,
            found="nonpositive cell area",
            expected="positive cell areas for centroid circles",
        )
    circles = gpd.GeoDataFrame(
        {"geometry": gpd.points_from_xy(coordinates[:, 0], coordinates[:, 1]), "radius": np.sqrt(areas / np.pi)},
        index=pd.Index(obs["instance_id"].array, name="instance_id"),
    )
    return ShapesModel.parse(circles, transformations={frame: Identity()})
