"""Build count-target points and literal cell contours in the DNB grid.

Points hold the eager int64 ``x, y`` grid coordinates of square bins, cells or
analysis observations, unchanged. Cell GEFs store each contour in
``cellBin/cellBorder`` as an ``(n_cells, 32, 2)`` int16 array of vertex offsets
relative to the cell centre, followed by paired ``(32767, 32767)`` padding;
vertices are centre plus offset. Contours are not repaired; the
``invalid_geometries`` policy keeps, drops or rejects invalid ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from spatialdata.models import PointsModel, ShapesModel

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.geometry import invalid_geometry_mask, raise_invalid_geometries

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import dask.dataframe as dd
    from numpy.typing import NDArray
    from spatialdata.transformations import BaseTransformation

    from spatialdata_io.readers._utils.geometry import InvalidGeometryPolicy

_PADDING = 32767
_MIN_VERTICES = 3


def make_points(
    coordinates: NDArray[np.int64], index: pd.Index, *, transformations: Mapping[str, BaseTransformation]
) -> dd.DataFrame:
    """Wrap eager count/analysis positions in the required points model without computing tasks."""
    # Keep numerical coordinates native: PointsModel's pandas path converts
    # Arrow columns to object arrays. Its ndarray boundary preserves int64.
    return PointsModel.parse(
        coordinates, annotation=pd.DataFrame(index=index), sort=False, transformations=dict(transformations)
    )


def make_polygons(  # noqa: PLR0913 - contour arrays, identity, policy and frame are separate inputs.
    coordinates: NDArray[np.int64],
    borders: NDArray[np.int16],
    index: pd.Index,
    *,
    path: Path,
    policy: InvalidGeometryPolicy,
    transformations: Mapping[str, BaseTransformation],
) -> gpd.GeoDataFrame:
    """Decode literal centroid-relative pairs and apply the selected topology policy."""
    context = ReaderErrorContext("stereoseq", "cell borders")
    if index.empty:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="empty cell population",
            expected="a nonempty polygon target (SpatialData requires nonempty shapes)",
        )
    # The producer stores centroid-relative pairs followed by paired SHRT_MAX
    # padding, rather than relative steps or an additional ring:
    # https://github.com/STOmics/geftools/blob/91998d84dd7a2f311304aca7f44122512bce8eae/mask.cpp#L86-L104
    padding_components = borders == _PADDING
    paired_padding = padding_components[:, :, 0] & padding_components[:, :, 1]
    vertex_counts = np.count_nonzero(~paired_padding, axis=1)
    malformed = (
        np.any(padding_components[:, :, 0] != padding_components[:, :, 1], axis=1)
        | np.any(np.maximum.accumulate(paired_padding, axis=1) != paired_padding, axis=1)
        | (vertex_counts < _MIN_VERTICES)
    )
    if malformed.any():
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"malformed paired padding/vertex counts for cells {index[malformed][:5].tolist()}",
            expected="at least three paired vertices followed only by paired (32767, 32767) padding",
        )

    # Widen signed offsets before addition. uint32 centers plus int16 offsets
    # fit exactly in int64 and in Shapely's float64 coordinate representation.
    vertices = (borders.astype(np.int64) + coordinates[:, None, :])[~paired_padding]
    ring_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(vertex_counts, dtype=np.int64)))
    polygons = shapely.from_ragged_array(
        shapely.GeometryType.POLYGON, vertices, (ring_offsets, np.arange(len(index) + 1, dtype=np.int64))
    )
    if policy != "keep":
        invalid = invalid_geometry_mask(polygons)
        if policy == "raise":
            raise_invalid_geometries(invalid, index.tolist(), context=context, path=path)
        elif invalid.any():
            polygons, index = polygons[~invalid], index[~invalid]
            if index.empty:
                raise ReaderFormatError(
                    context=context,
                    path=path,
                    found="all polygons dropped, leaving an empty element",
                    expected="at least one retained cell contour",
                )
    return ShapesModel.parse(gpd.GeoDataFrame(geometry=polygons, index=index), transformations=dict(transformations))
