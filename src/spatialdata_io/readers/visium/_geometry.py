"""Build the Visium spot circles and the three pixel frames that relate the images.

Spot centres stay in full-resolution image pixels, the frame of
``pxl_col_in_fullres`` (``x``) and ``pxl_row_in_fullres`` (``y``). Every
circle has the radius ``spot_diameter_fullres / 2``, the display size Space
Ranger draws, not a measured capture area.

The hires and lowres PNGs are the full-resolution image scaled by
``tissue_hires_scalef`` and ``tissue_lowres_scalef`` with aligned corners, so
one pure ``Scale`` per PNG relates the frames, with no translation::

    frame                       unit                    native elements
    <dataset_id>_pixels         full-resolution pixel   circles, full image
    <dataset_id>_hires_pixels   hires pixel             hires image
    <dataset_id>_lowres_pixels  lowres pixel            lowres image

    full-resolution -> hires:   Scale(tissue_hires_scalef)
    full-resolution -> lowres:  Scale(tissue_lowres_scalef)

The three names differ in their suffix, so they are distinct for every
``dataset_id`` (also ignoring case) and never equal an element name, which
ends in ``_image`` or is ``dataset_id`` itself. Circles carry all three
transformations; each PNG carries the identity into its own frame and the
inverse scale into the full-resolution frame. A calibrated original adds a
uniform scale from the full-resolution pixel frame to micrometres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
from spatialdata.models import ShapesModel
from spatialdata.transformations import BaseTransformation, Identity, Scale

from spatialdata_io.readers.visium._schemas import INSTANCE_KEY

if TYPE_CHECKING:
    import pandas as pd

    from spatialdata_io.readers.visium._metadata import ScaleFactors


@dataclass(frozen=True, slots=True)
class VisiumFrames:
    """Names of the three pixel frames and the scales from the full-resolution frame to the PNG frames."""

    fullres: str
    hires: str
    lowres: str
    hires_scale: float
    lowres_scale: float


def isotropic_scale(factor: float) -> Scale:
    """Apply one factor to both spatial axes."""
    return Scale([factor, factor], axes=("x", "y"))


def make_frames(dataset_id: str, scale_factors: ScaleFactors) -> VisiumFrames:
    """Name the pixel frames of one dataset and construct the full-resolution-to-PNG scales."""
    return VisiumFrames(
        fullres=f"{dataset_id}_pixels",
        hires=f"{dataset_id}_hires_pixels",
        lowres=f"{dataset_id}_lowres_pixels",
        hires_scale=scale_factors.hires,
        lowres_scale=scale_factors.lowres,
    )


def make_circles(
    positions: pd.DataFrame,
    *,
    radius: float,
    frames: VisiumFrames,
    microns_per_pixel: float | None = None,
) -> gpd.GeoDataFrame:
    """Construct barcode-indexed circles in full-resolution pixel coordinates."""
    circles = gpd.GeoDataFrame(
        {
            "geometry": gpd.points_from_xy(positions["pxl_col_in_fullres"], positions["pxl_row_in_fullres"]),
            "radius": np.full(len(positions), radius, dtype=np.float64),
        },
        index=positions.index.rename(INSTANCE_KEY),
    )

    transformations: dict[str, BaseTransformation] = {
        frames.fullres: Identity(),
        frames.hires: isotropic_scale(frames.hires_scale),
        frames.lowres: isotropic_scale(frames.lowres_scale),
    }
    if microns_per_pixel is not None:
        transformations["global"] = isotropic_scale(microns_per_pixel)
    return ShapesModel.parse(circles, transformations=transformations)
