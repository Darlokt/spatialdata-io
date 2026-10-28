"""Build Xenium boundary polygons and cell circles, applying the invalid-geometry policy.

Boundary files list one vertex per row: ``cell_id``, ``vertex_x``, ``vertex_y``
(micrometres) and, from XOA 2.0, ``label_id``. Rows of one polygon are
contiguous and each ring repeats its first vertex. Polygons are simplified for
visualization and are frequently not valid geometries; the masks in
``cells.zarr.zip`` are the exact segmentation.

Vertices are traced through boundary-pixel centres in pixel-index coordinates,
so they lie half a pixel up and left of the corner frame shared by masks, images,
centroids and transcripts. This was measured uniformly on bundles from XOA 1.0.1
to 4.0.0.19. Source vertices are kept; a ``Translation`` of half a pixel registers
them in the ``global`` micrometre frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity, Translation

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.geometry import invalid_geometry_mask, raise_invalid_geometries
from spatialdata_io.readers.xenium._schemas import (
    CELL_ID,
    FLOAT32,
    INT64,
    LABEL_ID,
    VERTICES,
    cast_column,
    id_index,
    id_type,
    normalize_ids,
    require_no_nulls,
    schema_error,
    validate_header,
)
from spatialdata_io.readers.xenium._tables import read_csv_table, read_header

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.geometry import InvalidGeometryPolicy
    from spatialdata_io.readers.xenium._cells import CellStore
    from spatialdata_io.readers.xenium._layout import TabularSource
    from spatialdata_io.readers.xenium._tables import Cells
    from spatialdata_io.readers.xenium._version import Family

type BoundaryKind = Literal["cell", "nucleus"]

# A LinearRing needs four coordinates including the repeated first vertex. Xenium
# Ranger imports occasionally emit shorter rings.
_MIN_RING_COORDINATES = 4


@dataclass(frozen=True, slots=True)
class Polygons:
    """Source polygons in file order, before the invalid-geometry policy.

    ``labels`` are mask labels when known and ``None`` for hexadecimal-ID nuclei
    without ``cells.zarr.zip``. ``geometries`` is ``None`` where a ring has fewer
    than four coordinates (``degenerate``).
    """

    cell_ids: pd.Index
    labels: NDArray[np.uint32] | None
    geometries: NDArray[np.object_]
    degenerate: NDArray[np.bool_]
    source: TabularSource


@dataclass(frozen=True, slots=True)
class _Vertices:
    """Projected vertex rows of one boundary file."""

    ids: pd.Index
    labels: NDArray[np.int64] | None
    coordinates: NDArray[np.float64]


def _read_vertices(source: TabularSource, family: Family, kind: BoundaryKind) -> _Vertices:
    path, artifact = source.file, f"{kind} boundaries"
    names = pq.read_schema(path).names if source.kind == "parquet" else read_header(source)
    validate_header(names, (CELL_ID, *VERTICES), path=path, artifact=artifact)
    columns = [CELL_ID, *VERTICES, *((LABEL_ID,) if LABEL_ID in names else ())]
    types = {CELL_ID: id_type(family), VERTICES[0]: FLOAT32, VERTICES[1]: FLOAT32, LABEL_ID: INT64}
    table = (
        pq.read_table(path, columns=columns)
        if source.kind == "parquet"
        else read_csv_table(source, names, types, columns=columns)
    )
    require_no_nulls(table, path=path, artifact=artifact)

    ids = id_index(normalize_ids(table[CELL_ID], family, path=path, artifact=artifact))
    # float32 vertices are exact in the float64 coordinates Shapely uses.
    coordinates = np.column_stack(
        [
            np.asarray(
                cast_column(table[name], FLOAT32, name, path=path, artifact=artifact).to_numpy(), dtype=np.float64
            )
            for name in VERTICES
        ]
    )
    labels = None
    if LABEL_ID in columns:
        labels = np.asarray(cast_column(table[LABEL_ID], INT64, LABEL_ID, path=path, artifact=artifact).to_numpy())
    return _Vertices(ids, labels, coordinates)


def read_polygons(source: TabularSource, family: Family, kind: BoundaryKind, store: CellStore | None) -> Polygons:
    """Group contiguous vertex rows into polygons and check them against the cell store.

    Raises
    ------
    ReaderFormatError
        If rows of one polygon are not contiguous, rings are not closed or not
        finite, a ``label_id`` polygon spans several cells, polygon-set nuclei lack
        ``label_id``, labels disagree with ``cells.zarr.zip``, or IDs are unknown.
    """
    path, artifact = source.file, f"{kind} boundaries"
    vertices = _read_vertices(source, family, kind)
    if family == "polygon_sets" and kind == "nucleus" and vertices.labels is None:
        raise schema_error(
            path,
            artifact,
            found="nucleus boundaries without label_id (early XOA 2.0 builds)",
            expected="label_id to separate the nuclei of multinucleate cells; use include_nucleus_boundaries=False",
        )
    if not np.isfinite(vertices.coordinates).all():
        raise schema_error(path, artifact, found="nonfinite vertices", expected="finite micrometre vertices")

    # A polygon is one run of equal keys: label_id when present, otherwise cell_id.
    id_codes, unique_ids = vertices.ids.factorize()
    keys = id_codes if vertices.labels is None else vertices.labels
    starts = np.flatnonzero(np.concatenate([[True], keys[1:] != keys[:-1]])) if len(keys) else np.empty(0, np.intp)
    if len(np.unique(keys[starts])) != len(starts):
        raise schema_error(path, artifact, found="interleaved polygon rows", expected="contiguous rows per polygon")
    lengths = np.diff(np.append(starts, len(keys)))
    if (id_codes != np.repeat(id_codes[starts], lengths)).any():
        raise schema_error(path, artifact, found="a polygon spanning several cells", expected="one cell per label_id")
    if not (vertices.coordinates[starts] == vertices.coordinates[starts + lengths - 1]).all():
        raise schema_error(path, artifact, found="open rings", expected="rings repeating their first vertex")

    degenerate = lengths < _MIN_RING_COORDINATES
    geometries = np.full(len(starts), None, dtype=object)
    if not degenerate.all():
        built = ~degenerate
        offsets = np.concatenate([[0], np.cumsum(lengths[built])])
        geometries[built] = shapely.from_ragged_array(
            shapely.GeometryType.POLYGON,
            vertices.coordinates[np.repeat(built, lengths)],
            (offsets, np.arange(built.sum() + 1)),
        )

    cell_ids = pd.Index(unique_ids)[id_codes[starts]]
    file_labels = None if vertices.labels is None else vertices.labels[starts]
    if store is not None:
        labels = _store_labels(file_labels, cell_ids, kind, store, path)
    elif file_labels is not None:
        labels = _uint32_labels(file_labels, path, artifact)
    elif family == "integer_ids":
        # Integer cell IDs are their own mask labels; nuclei carry their cell's label.
        labels = _uint32_labels(cell_ids.to_numpy(dtype=np.int64), path, artifact)
    else:
        labels = None
    return Polygons(cell_ids, labels, geometries, degenerate, source)


def _uint32_labels(values: NDArray[np.int64], path: Path, artifact: str) -> NDArray[np.uint32]:
    if (values <= 0).any() or values.max(initial=0) > np.iinfo(np.uint32).max:
        raise schema_error(path, artifact, found="nonpositive or oversized labels", expected="uint32 labels from 1")
    return values.astype(np.uint32)


def _store_labels(
    file_labels: NDArray[np.int64] | None,
    cell_ids: pd.Index,
    kind: BoundaryKind,
    store: CellStore,
    path: Path,
) -> NDArray[np.uint32]:
    """Return polygon mask labels whose owning cells agree with ``cells.zarr.zip``."""
    artifact = f"{kind} boundaries"
    rows = store.cell_ids.get_indexer(cell_ids)
    if (rows < 0).any():
        raise schema_error(path, artifact, found="cell IDs absent from cells.zarr.zip", expected="known cell IDs")
    if file_labels is None:
        # Integer and hexadecimal families: a nucleus carries its cell's label.
        return store.cell_labels[rows]

    # Polygon sets: label = polygon index + 1 of the named set, owned by cell_index.
    labels = _uint32_labels(file_labels, path, artifact)
    if kind == "cell":
        agrees = labels == store.cell_labels[rows]
    elif store.nuclei is None:
        raise schema_error(path, artifact, found="no nucleus polygon set in cells.zarr.zip", expected="a nucleus set")
    else:
        owners = store.nuclei.cells
        inside = labels <= len(owners)
        agrees = inside & (owners[np.where(inside, labels - 1, 0)] == rows)
    if not agrees.all():
        raise schema_error(
            path,
            artifact,
            found="label_id values owned by other cells in cells.zarr.zip",
            expected="label_id = polygon index + 1 of the stored set, owned by the polygon's cell",
        )
    return labels


def apply_policy(polygons: Polygons, policy: InvalidGeometryPolicy, kind: BoundaryKind) -> NDArray[np.bool_]:
    """Return which polygons remain; rings too short to build count as invalid entities.

    Raises
    ------
    ReaderFormatError
        Under ``"raise"`` for any invalid or degenerate polygon, and under
        ``"keep"`` for degenerate rings, which cannot be represented.
    """
    built = ~polygons.degenerate
    invalid = polygons.degenerate.copy()
    invalid[built] = invalid_geometry_mask(polygons.geometries[built])
    context = ReaderErrorContext("xenium", f"{kind} boundaries")
    # Report the identifiers that index the returned element.
    labels = polygons.labels if kind == "nucleus" else None
    identifiers = polygons.cell_ids.tolist() if labels is None else labels.tolist()
    if policy == "raise":
        raise_invalid_geometries(invalid, identifiers, context=context, path=polygons.source.file)
    if policy == "keep" and polygons.degenerate.any():
        examples = [identifiers[index] for index in np.flatnonzero(polygons.degenerate)[:10]]
        raise ReaderFormatError(
            context=context,
            path=polygons.source.file,
            found=f"{int(polygons.degenerate.sum())} rings with fewer than four coordinates, e.g. {examples!r}",
            expected="buildable polygons; invalid_geometries='drop' removes them as invalid entities",
        )
    return built if policy == "keep" else ~invalid


def _parse(frame: gpd.GeoDataFrame, source: TabularSource, kind: BoundaryKind, pixel_size: float) -> gpd.GeoDataFrame:
    if frame.empty:
        raise schema_error(
            source.file,
            f"{kind} boundaries",
            found="no polygons to return",
            expected=f"at least one polygon; use include_{kind}_boundaries=False",
        )
    # Register pixel-index vertices in the corner frame without changing them.
    half = pixel_size / 2
    return ShapesModel.parse(frame, transformations={"global": Translation([half, half], axes=("x", "y"))})


def make_cell_polygons(polygons: Polygons, kept: NDArray[np.bool_], pixel_size: float) -> gpd.GeoDataFrame:
    """Index cell polygons by ``cell_id`` in the registered micrometre frame."""
    index = polygons.cell_ids[kept].rename(CELL_ID)
    frame = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygons.geometries[kept], index=index))
    return _parse(frame, polygons.source, "cell", pixel_size)


def make_nucleus_polygons(polygons: Polygons, kept: NDArray[np.bool_], pixel_size: float) -> gpd.GeoDataFrame:
    """Index nucleus polygons by nucleus mask label, with their owning ``cell_id``."""
    if polygons.labels is None:
        raise schema_error(
            polygons.source.file,
            "nucleus boundaries",
            found="hexadecimal-ID nuclei without cells.zarr.zip",
            expected="seg_mask_value labels from cells.zarr.zip; use include_nucleus_boundaries=False",
        )
    index = pd.Index(polygons.labels[kept], name=LABEL_ID)
    cell_ids = pd.Series(polygons.cell_ids[kept].array, index=index)
    frame = gpd.GeoDataFrame({CELL_ID: cell_ids}, geometry=gpd.GeoSeries(polygons.geometries[kept], index=index))
    return _parse(frame, polygons.source, "nucleus", pixel_size)


def make_circles(cells: Cells, kept: NDArray[np.bool_]) -> gpd.GeoDataFrame:
    """Return centroid circles whose area equals the exact mask ``cell_area``."""
    centres = cells.frame[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)[kept]
    radius = np.sqrt(cells.frame["cell_area"].to_numpy(dtype=np.float64)[kept] / np.pi)
    index = cells.ids[kept].rename(CELL_ID)
    geometry = gpd.GeoSeries(shapely.points(centres), index=index)
    frame = gpd.GeoDataFrame({"radius": pd.Series(radius, index=index)}, geometry=geometry)
    return ShapesModel.parse(frame, transformations={"global": Identity()})
