"""Exact bin footprints and projected Space Ranger polygon ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyogrio.errors import DataSourceError, GeometryError  # type: ignore[import-untyped]  # Pyogrio has no type marker.
from shapely.errors import GEOSException
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers.visium_hd._transforms import SlideFrame, grid_to_global, image_to_global

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.paths import ArtifactFile

MAX_CELL_ID = 999_999_999


@dataclass(frozen=True, slots=True)
class BoundaryData:
    """Source-order numeric IDs, global polygons and optional vendor centroids."""

    shapes: gpd.GeoDataFrame
    centroids: NDArray[np.float64] | None


def _validate_source(source: gpd.GeoDataFrame, path: ArtifactFile, context: ReaderErrorContext) -> None:
    """Validate external IDs and ring representation independently of topology policy."""

    def fail(found: str, expected: str) -> ReaderFormatError:
        return ReaderFormatError(context=context, found=found, expected=expected, path=path)

    if "cell_id" not in source:
        msg = "missing cell_id"
        raise fail(msg, "unique nonnull positive integer cell_id properties")

    ids = source["cell_id"]
    if ids.isna().any() or not pd.api.types.is_integer_dtype(ids.dtype):
        msg = "null or non-integer cell_id"
        raise fail(msg, "unique integer cell_id values from 1 through 999999999")
    if ((ids <= 0) | (ids > MAX_CELL_ID)).any() or ids.duplicated().any():
        msg = "duplicate or out-of-range cell_id"
        raise fail(msg, "unique integer cell_id values from 1 through 999999999")

    if source.geometry.isna().any() or source.geometry.is_empty.any() or (source.geometry.geom_type != "Polygon").any():
        msg = "missing, empty or non-Polygon geometry"
        raise fail(msg, "one nonempty Polygon per feature")
    if (shapely.get_num_interior_rings(source.geometry.to_numpy()) != 0).any():
        msg = "Polygon interior rings"
        raise fail(msg, "one exterior ring without holes")
    if shapely.has_z(source.geometry.to_numpy()).any() or shapely.has_m(source.geometry.to_numpy()).any():
        msg = "non-planar polygon coordinates"
        raise fail(msg, "two-dimensional polygon vertices")

    coordinates = shapely.get_coordinates(source.geometry.to_numpy())
    if not np.isfinite(coordinates).all():
        msg = "nonfinite polygon coordinates"
        raise fail(msg, "finite polygon coordinates")


def read_boundaries(path: ArtifactFile, frame: SlideFrame, *, kind: Literal["cell", "nucleus"]) -> BoundaryData:
    """Read projected GeoJSON without Python feature dictionaries or geometry repair.

    GDAL retains source order and ring vertices. Only Polygon exteriors are
    supported; malformed geometry is rejected independently of validity policy.
    """
    context = ReaderErrorContext(reader="visium_hd", artifact=f"{kind} boundaries")

    def fail(found: str, expected: str) -> ReaderFormatError:
        return ReaderFormatError(context=context, found=found, expected=expected, path=path)

    try:
        # Preserve standard I/O errors before GDAL wraps failures as driver errors.
        path.stat()
        source = gpd.read_file(path, engine="pyogrio", columns=["cell_id", f"{kind}_centroid"], on_invalid="raise")
    except (DataSourceError, GeometryError, GEOSException) as error:
        msg = "malformed GeoJSON or polygon ring"
        raise fail(msg, "a FeatureCollection of closed Polygon rings") from error
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise

    _validate_source(source, path, context)

    coordinates = shapely.get_coordinates(source.geometry.to_numpy())
    transformed = image_to_global(frame, coordinates)
    geometry = shapely.set_coordinates(source.geometry.to_numpy(), transformed)
    index = pd.Index(source["cell_id"].to_numpy(dtype=np.int64), name="cell_id")
    shapes = gpd.GeoDataFrame({"geometry": geometry}, index=index)

    centroid_name = f"{kind}_centroid"
    centroids = None
    if centroid_name in source:
        try:
            centroids = np.array(source[centroid_name].tolist(), dtype=np.float64)
        except (TypeError, ValueError) as error:
            msg = "malformed centroid property"
            raise fail(msg, "one finite [x, y] centroid per polygon") from error
        if centroids.shape != (len(source), 2) or not np.isfinite(centroids).all():
            msg = "missing or nonfinite centroid property"
            raise fail(msg, "one finite [x, y] centroid per polygon")
        centroids = image_to_global(frame, centroids)

    return BoundaryData(ShapesModel.parse(shapes, transformations={"global": Identity()}), centroids)


def bin_squares(
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
    size_um: int,
    frame: SlideFrame,
    index: pd.Index,
) -> gpd.GeoDataFrame:
    """Build nominal full squares, including the documented capture-edge overhang."""
    if len(rows) == 0:
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "bin squares"),
            found="empty bin population",
            expected="at least one bin for a SpatialData shapes element",
        )

    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    geometries = np.empty(len(rows), dtype=object)

    # The eager geometry vector is required; transient corner buffers remain bounded.
    for start in range(0, len(rows), 65_536):
        stop = start + 65_536
        origin = np.column_stack((cols[start:stop], rows[start:stop])).astype(np.float64) * size_um
        vertices = origin[:, None, :] + size_um * corners[None, :, :]
        transformed = grid_to_global(frame, vertices.reshape(-1, 2)).reshape(-1, 5, 2)
        geometries[start:stop] = shapely.polygons(transformed)

    return ShapesModel.parse(
        gpd.GeoDataFrame({"geometry": geometries}, index=index), transformations={"global": Identity()}
    )
