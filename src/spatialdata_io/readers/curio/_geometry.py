"""Calibrate Seeker bead positions and build one display circle per bead.

The primary pipeline stores bead XY in ``obsm["X_spatial"]`` as an
``(n_obs, 2)`` array of Seeker pixels in ``(x, y)`` order, and some exports
repeat it in ``obsm["spatial"]``. Both must be finite real arrays; a repeat
must equal ``X_spatial`` exactly. Shapely stores the circle centres as
float64. Other coordinate profiles (uppercase ``X_SPATIAL``,
Trekker, merged or reprocessed objects) are not supported.

Takara's coordinate-unit FAQ gives approximately 0.647 µm per Seeker pixel
(https://help.takarabio.com/36168/kb/article/137541). The calibration is
approximate and registers no image. The reader keeps the source pixels, with
their dtype, origin and orientation, as ``obsm["spatial_pixels"]``, writes
float64 micrometres to ``obsm["spatial"]`` and describes both in
``uns["spatialdata_io_positions"]``. The circles stay in pixels and reach the
``global`` micrometre frame through ``Scale(0.647)``. Their radius is half the
nominal 10 µm bead diameter stated in Takara's Seeker FAQ
(https://www.takarabio.com/learning-centers/spatial-omics/seeker-resources/seeker-faqs),
expressed in Seeker pixels (5 / 0.647, about 7.73 pixels). Individual beads vary
around this nominal size.
"""

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

MICROMETRES_PER_PIXEL = 0.647
BEAD_DIAMETER_MICROMETRES = 10.0
BEAD_RADIUS_PIXELS = BEAD_DIAMETER_MICROMETRES / 2 / MICROMETRES_PER_PIXEL
_CALIBRATION = (
    f"Approximate manufacturer calibration: {MICROMETRES_PER_PIXEL} micrometres per Seeker primary pixel; "
    "Takara coordinate-unit FAQ https://help.takarabio.com/36168/kb/article/137541"
)


def read_shapes(table: ad.AnnData, path: ArtifactFile) -> gpd.GeoDataFrame:
    """Move ``X_spatial`` to ``spatial_pixels``, add micrometre positions and build bead circles."""
    context = ReaderErrorContext("curio", "spatial coordinates")
    if "spatial_pixels" in table.obsm or POSITIONS_KEY in table.uns or "X_SPATIAL" in table.obsm:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="occupied coordinate destination or unsupported uppercase profile",
            expected="Seeker-primary X_spatial with optional agreeing spatial and unoccupied output destinations",
        )

    xy = _validate_xy(table.obsm.get("X_spatial"), table.n_obs, context, path)
    repeat = table.obsm.get("spatial")
    if repeat is not None and not np.array_equal(xy, _validate_xy(repeat, table.n_obs, context, path)):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="disagreeing spatial arrays",
            expected="X_spatial and optional spatial with exactly equal coordinates",
        )

    with np.errstate(over="ignore"):
        physical = np.multiply(xy, MICROMETRES_PER_PIXEL, dtype=np.float64)
    if not np.isfinite(physical).all():
        raise ReaderFormatError(
            context=context,
            path=path,
            found="coordinates overflow float64",
            expected="finite float64 physical XY coordinates",
        )

    table.obsm["spatial_pixels"] = table.obsm.pop("X_spatial")
    table.obsm["spatial"] = physical
    table.uns[POSITIONS_KEY] = _positions_metadata()

    shapes = gpd.GeoDataFrame(
        {"geometry": gpd.points_from_xy(xy[:, 0], xy[:, 1]), "radius": BEAD_RADIUS_PIXELS},
        # A separate Index object: ShapesModel keeps it, and its mutable name must not alias obs_names.
        index=table.obs_names.copy(),
    )
    scale = Scale([MICROMETRES_PER_PIXEL, MICROMETRES_PER_PIXEL], axes=("x", "y"))
    return ShapesModel.parse(shapes, transformations={"global": scale})


def _validate_xy(
    xy: object, n_obs: int, context: ReaderErrorContext, path: ArtifactFile
) -> NDArray[np.integer | np.floating]:
    """Require finite real ``(n_obs, 2)`` coordinates."""
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
    return xy


def _positions_metadata() -> PositionsMetadata:
    """Describe the native pixel and calibrated micrometre positions."""
    return {
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
                "calibration": {"global": _CALIBRATION},
            },
        },
    }
