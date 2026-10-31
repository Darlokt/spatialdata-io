"""Validate Seeker primary pixels and construct calibrated bead display geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
from spatialdata.models import ShapesModel
from spatialdata.transformations import Scale

from spatialdata_io._positions import POSITIONS_KEY, PositionsMetadata
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    import anndata as ad
    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.paths import ArtifactFile


def _validate_xy(
    xy: object, n_obs: int, context: ReaderErrorContext, path: ArtifactFile
) -> NDArray[np.integer | np.floating]:
    """Validate one borrowed embedding without converting its native buffer."""
    if (
        not isinstance(xy, np.ndarray)
        or xy.shape != (n_obs, 2)
        or xy.dtype.kind not in "iuf"
        or not np.isfinite(xy).all()
    ):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="missing or invalid spatial array",
            expected="finite real numeric coordinates of shape (n_obs, 2), ordered (x, y)",
        )

    # Shapely stores float64 coordinates. Reject a lossy conversion before geometry construction.
    if xy.dtype.kind in "iu":
        candidates = xy[(xy > 2**53) | (xy < -(2**53))]
        loss = any(int(value) != int(float(value)) for value in candidates)
    elif xy.dtype.itemsize > np.dtype(np.float64).itemsize:
        with np.errstate(over="ignore"):
            loss = not np.array_equal(xy, xy.astype(np.float64))
    else:
        loss = False

    if loss:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="spatial coordinates lose precision in float64 geometry",
            expected="coordinates losslessly representable as float64",
        )
    return xy


def _read_shapes(table: ad.AnnData, path: ArtifactFile) -> gpd.GeoDataFrame:
    """Normalize primary-pipeline pixels and construct radius-10 pixel bead geometry."""
    context = ReaderErrorContext("curio", "spatial coordinates")
    if "spatial_pixels" in table.obsm or POSITIONS_KEY in table.uns or "X_SPATIAL" in table.obsm:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="occupied coordinate destination or unsupported uppercase profile",
            expected="Seeker-primary X_spatial with optional agreeing spatial and unoccupied output destinations",
        )
    xy = _validate_xy(table.obsm.get("X_spatial"), table.n_obs, context, path)
    duplicate = table.obsm.get("spatial")
    if duplicate is not None:
        duplicate = _validate_xy(duplicate, table.n_obs, context, path)
        # Both sources are exactly float64-representable, so mixed-dtype
        # promotion cannot hide distinct integer coordinates during comparison.
        if not np.array_equal(xy, duplicate):
            raise ReaderFormatError(
                context=context,
                path=path,
                found="disagreeing spatial arrays",
                expected="X_spatial and optional spatial with exactly equal coordinates",
            )

    # The manufacturer calibrates primary-pipeline pixels approximately; retain
    # their exact dtype, orientation and origin before creating physical XY once.
    # Finite source coordinates are losslessly representable in float64 and
    # 0.647 is below one, so multiplication cannot produce nonfinite positions.
    physical = np.multiply(xy, 0.647, dtype=np.float64)
    table.obsm["spatial_pixels"] = table.obsm.pop("X_spatial")
    table.obsm["spatial"] = physical
    positions: PositionsMetadata = {
        "schema_version": 1,
        "coordinates": {
            "spatial_pixels": {
                "axes": ["x", "y"],
                "units": "pixel",
                "frame": "seeker_pixels",
                "position": "source bead reference position",
            },
            "spatial": {
                "axes": ["x", "y"],
                "units": "micrometer",
                "frame": "global",
                "position": "source bead reference position",
                "calibration": {
                    "global": "Approximate manufacturer calibration: 0.647 micrometres per Seeker primary pixel; "
                    "Takara coordinate-unit FAQ https://help.takarabio.com/36168/kb/article/137541"
                },
            },
        },
    }
    table.uns[POSITIONS_KEY] = positions
    shapes = gpd.GeoDataFrame(
        {"geometry": gpd.points_from_xy(xy[:, 0], xy[:, 1]), "radius": 10.0},
        index=table.obs_names.copy(),
    )
    return ShapesModel.parse(shapes, transformations={"global": Scale([0.647, 0.647], axes=("x", "y"))})
