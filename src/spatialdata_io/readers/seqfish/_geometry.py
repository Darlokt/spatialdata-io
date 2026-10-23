"""Construct physical circles and explicitly identified pixel boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.metadata import read_json_object
from spatialdata_io.readers.seqfish._schemas import boundary_features, schema_error

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata.transformations import Scale

    from spatialdata_io.readers._utils.paths import ArtifactFile

_MAX_BOUNDARY_BYTES = 64 * 1024 * 1024


def read_boundaries(path: ArtifactFile, *, instance_ids: pd.Index, roi: str, scale: Scale) -> gpd.GeoDataFrame:
    """Read eager GeoJSON polygons and align their independently encoded IDs."""
    document = read_json_object(
        path, max_bytes=_MAX_BOUNDARY_BYTES, context=ReaderErrorContext("seqfish", "cell boundaries")
    )
    features, identifiers = boundary_features(document, path)
    frame = gpd.GeoDataFrame.from_features(features)
    geometry = frame.geometry
    if (
        geometry.isna().any()
        or geometry.is_empty.any()
        or not geometry.is_valid.all()
        or not geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all()
        or geometry.has_z.any()
    ):
        raise schema_error(path, found="invalid, empty or nonplanar boundaries", expected="valid finite 2D polygons")
    frame.index = pd.Index(identifiers, name="instance_id")
    if len(frame) != len(instance_ids) or (frame.index.get_indexer(instance_ids) < 0).any():
        raise schema_error(path, found="boundary/cell ID disagreement", expected="exactly the table cell IDs")
    if not frame.index.equals(instance_ids):
        frame = frame.loc[instance_ids]
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
