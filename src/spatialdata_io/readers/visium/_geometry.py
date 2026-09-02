"""Visium spot geometry and coordinate transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity, Scale

if TYPE_CHECKING:
    import pandas as pd
    from spatialdata.transformations import BaseTransformation

    from spatialdata_io.readers.visium._metadata import ScaleFactors


@dataclass(frozen=True, slots=True)
class VisiumTransforms:
    """Named full-resolution and downsampled coordinate transformations."""

    fullres: BaseTransformation
    hires: BaseTransformation
    lowres: BaseTransformation


def make_transforms(scale_factors: ScaleFactors) -> VisiumTransforms:
    """Construct full-resolution-to-downsampled transformations."""
    return VisiumTransforms(
        fullres=Identity(),
        hires=Scale(np.asarray([scale_factors.hires, scale_factors.hires]), axes=("x", "y")),
        lowres=Scale(np.asarray([scale_factors.lowres, scale_factors.lowres]), axes=("x", "y")),
    )


def make_circles(
    positions: pd.DataFrame,
    *,
    dataset_id: str,
    radius: float,
    transforms: VisiumTransforms,
) -> gpd.GeoDataFrame:
    """Construct barcode-indexed circles in full-resolution pixel coordinates."""
    circles = gpd.GeoDataFrame(
        {
            "geometry": gpd.points_from_xy(
                positions["pxl_col_in_fullres"],
                positions["pxl_row_in_fullres"],
            ),
            "radius": np.full(len(positions), radius, dtype=np.float64),
        },
        index=positions.index.copy(),
    )
    circles.index.name = "spot_id"
    return ShapesModel.parse(
        circles,
        transformations={
            dataset_id: transforms.fullres,
            f"{dataset_id}_downscaled_hires": transforms.hires,
            f"{dataset_id}_downscaled_lowres": transforms.lowres,
        },
    )
