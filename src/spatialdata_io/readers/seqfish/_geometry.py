"""Construct physical circles and explicitly identified pixel boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.geometry import geojson_polygons, invalid_geometry_mask, raise_invalid_geometries
from spatialdata_io.readers._utils.metadata import read_json_object
from spatialdata_io.readers.seqfish._schemas import boundary_features, schema_error

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata.transformations import Scale

    from spatialdata_io.readers._utils.geometry import InvalidGeometryPolicy
    from spatialdata_io.readers._utils.paths import ArtifactFile

_CONTEXT = ReaderErrorContext("seqfish", "cell boundaries")


def read_boundaries(
    path: ArtifactFile, *, instance_ids: pd.Index, roi: str, scale: Scale, policy: InvalidGeometryPolicy
) -> gpd.GeoDataFrame:
    """Read eager GeoJSON polygons and align their independently encoded IDs."""
    document = read_json_object(path, context=_CONTEXT)
    features, identifiers = boundary_features(document, path)
    geometries = geojson_polygons([feature["geometry"] for feature in features], context=_CONTEXT, path=path)
    # Keep arbitrary annotation IDs exact, including nullable integers, rather
    # than allowing pandas to infer a lossy floating column from JSON values.
    frame = gpd.GeoDataFrame(
        [
            {"geometry": geometry, **(feature["properties"] or {})}
            for geometry, feature in zip(geometries, features, strict=True)
        ],
        dtype=object,
    )
    frame.index = pd.Index(identifiers, name="instance_id")
    if len(frame) != len(instance_ids) or (frame.index.get_indexer(instance_ids) < 0).any():
        raise schema_error(path, found="boundary/cell ID disagreement", expected="exactly the table cell IDs")
    if not frame.index.equals(instance_ids):
        frame = frame.loc[instance_ids]
    if policy != "keep":
        invalid = invalid_geometry_mask(frame.geometry.to_numpy())
        if policy == "raise":
            raise_invalid_geometries(invalid, frame.index.tolist(), context=_CONTEXT, path=path)
        elif invalid.any():
            frame = frame.loc[~invalid]
            if frame.empty:
                raise schema_error(
                    path,
                    found="all boundaries dropped, leaving an empty element",
                    expected="at least one retained cell boundary",
                )
    return ShapesModel.parse(frame, transformations={roi: scale})


def make_circles(table: AnnData, *, roi: str) -> gpd.GeoDataFrame:
    """Use physical centroids and square-micrometer areas for cell circles."""
    # The table parser owns eager, finite, float64 XY coordinates and areas.
    coordinates = cast("np.ndarray", table.obsm["spatial"])
    # The reader-created table is eager; AnnData's backed Dataset2D alternative
    # does not apply at this private producer boundary.
    obs = cast("pd.DataFrame", table.obs)
    areas = obs["area"].to_numpy(dtype=np.float64)
    if (areas <= 0).any():
        message = "seqfish cell circles require positive physical cell areas."
        raise ValueError(message)
    frame = gpd.GeoDataFrame(
        {"geometry": gpd.points_from_xy(coordinates[:, 0], coordinates[:, 1]), "radius": np.sqrt(areas / np.pi)},
        index=pd.Index(obs["instance_id"].to_numpy(), name="instance_id"),
    )
    return ShapesModel.parse(frame, transformations={roi: Identity()})
