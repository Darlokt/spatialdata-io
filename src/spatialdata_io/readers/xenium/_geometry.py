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
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import shapely
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity, Translation

from spatialdata_io.readers._utils.csv import csv_errors
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.geometry import invalid_geometry_mask, raise_invalid_geometries
from spatialdata_io.readers.xenium._schemas import (
    CELL_ID,
    LABEL_ID,
    VERTICES,
    cast_column,
    id_index,
    normalize_ids,
    require_no_nulls,
    validate_header,
)
from spatialdata_io.readers.xenium._tables import open_twin, read_header

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.geometry import InvalidGeometryPolicy
    from spatialdata_io.readers._utils.paths import ArtifactFile
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
    than four coordinates (``degenerate``). ``path`` and ``context`` identify the
    source in later policy and model errors.
    """

    kind: BoundaryKind
    cell_ids: pd.Index
    labels: NDArray[np.uint32] | None
    geometries: NDArray[np.object_]
    degenerate: NDArray[np.bool_]
    path: ArtifactFile
    context: ReaderErrorContext


@dataclass(frozen=True, slots=True)
class _Vertices:
    """Projected vertex rows of one boundary file."""

    ids: pd.Index
    labels: NDArray[np.uint32] | None
    coordinates: NDArray[np.float64]


def _read_vertices(source: TabularSource, context: ReaderErrorContext) -> _Vertices:
    path, types = source.file, source.types
    names = pq.read_schema(path).names if source.kind == "parquet" else read_header(source, context=context)
    validate_header(names, (CELL_ID, *VERTICES), path=path, context=context)
    columns = [CELL_ID, *VERTICES, *((LABEL_ID,) if LABEL_ID in names else ())]
    if source.kind == "parquet":
        table = pq.read_table(path, columns=columns)
    else:
        with csv_errors(context=context, path=path), open_twin(source) as stream:
            table = pacsv.read_csv(
                stream,
                convert_options=pacsv.ConvertOptions(
                    column_types=types, include_columns=columns, null_values=[], strings_can_be_null=False
                ),
            )
    require_no_nulls(table, path=path, context=context)

    ids = id_index(normalize_ids(table[CELL_ID], types.field(CELL_ID).type, path=path, context=context))
    # float32 vertices are exact in the float64 coordinates Shapely uses.
    coordinates = np.column_stack(
        [
            np.asarray(
                cast_column(table[name], types.field(name).type, name, path=path, context=context).to_numpy(),
                dtype=np.float64,
            )
            for name in VERTICES
        ]
    )
    labels = None
    if LABEL_ID in columns:
        kind = types.field(LABEL_ID).type
        labels = np.asarray(cast_column(table[LABEL_ID], kind, LABEL_ID, path=path, context=context).to_numpy())
    return _Vertices(ids, labels, coordinates)


def _runs(vertices: _Vertices, path: Path, context: ReaderErrorContext) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return the first row and the row count of every polygon, checking contiguity and closure."""

    def error(*, found: str, expected: str) -> ReaderFormatError:
        return ReaderFormatError(context=context, path=path, found=found, expected=expected)

    if not np.isfinite(vertices.coordinates).all():
        raise error(found="nonfinite vertices", expected="finite micrometre vertices")
    # A polygon is one run of equal keys: label_id when present, otherwise cell_id.
    id_codes = vertices.ids.factorize()[0]
    keys = id_codes if vertices.labels is None else vertices.labels
    starts = np.flatnonzero(np.concatenate([[True], keys[1:] != keys[:-1]])) if len(keys) else np.empty(0, np.intp)
    if len(np.unique(keys[starts])) != len(starts):
        raise error(found="interleaved polygon rows", expected="contiguous rows per polygon")
    lengths = np.diff(np.append(starts, len(keys)))
    if (id_codes != np.repeat(id_codes[starts], lengths)).any():
        raise error(found="a polygon spanning several cells", expected="one cell per label_id")
    if not (vertices.coordinates[starts] == vertices.coordinates[starts + lengths - 1]).all():
        raise error(found="open rings", expected="rings repeating their first vertex")
    return starts, lengths


def read_polygons(
    source: TabularSource,
    family: Family,
    kind: BoundaryKind,
    store: CellStore | None,
    *,
    context: ReaderErrorContext,
) -> Polygons:
    """Group contiguous vertex rows into polygons and check them against the cell store.

    Raises
    ------
    ReaderFormatError
        If rows of one polygon are not contiguous, rings are not closed or not
        finite, a ``label_id`` polygon spans several cells, polygon-set nuclei lack
        ``label_id``, labels disagree with ``cells.zarr.zip``, or IDs are unknown.
    """
    path = source.file
    vertices = _read_vertices(source, context)
    if family == "polygon_sets" and kind == "nucleus" and vertices.labels is None:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="nucleus boundaries without label_id (early XOA 2.0 builds)",
            expected="label_id to separate the nuclei of multinucleate cells; use include_nucleus_boundaries=False",
        )
    starts, lengths = _runs(vertices, path, context)

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

    cell_ids = vertices.ids[starts]
    file_labels = None if vertices.labels is None else vertices.labels[starts]
    if store is not None:
        labels = _store_labels(file_labels, cell_ids, kind, store, path=path, context=context)
    elif file_labels is not None:
        labels = _uint32_labels(file_labels, path, context)
    elif family == "integer_ids":
        # Integer cell IDs are their own mask labels; nuclei carry their cell's label.
        labels = _uint32_labels(cell_ids.to_numpy(dtype=np.uint64), path, context)
    else:
        labels = None
    return Polygons(kind, cell_ids, labels, geometries, degenerate, path, context)


def _uint32_labels(
    values: NDArray[np.uint32] | NDArray[np.uint64], path: Path, context: ReaderErrorContext
) -> NDArray[np.uint32]:
    labels = cast_column(pa.chunked_array([values]), pa.uint32(), LABEL_ID, path=path, context=context).to_numpy()
    if (labels == 0).any():
        raise ReaderFormatError(
            context=context, path=path, found="nonpositive or oversized labels", expected="uint32 labels from 1"
        )
    return labels


def _store_labels(  # noqa: PLR0913, RUF100 - Labels, owners and the error origin.
    file_labels: NDArray[np.uint32] | None,
    cell_ids: pd.Index,
    kind: BoundaryKind,
    store: CellStore,
    *,
    path: Path,
    context: ReaderErrorContext,
) -> NDArray[np.uint32]:
    """Return polygon mask labels whose owning cells agree with ``cells.zarr.zip``."""
    rows = store.cell_ids.get_indexer(cell_ids)
    if (rows < 0).any():
        raise ReaderFormatError(
            context=context, path=path, found="cell IDs absent from cells.zarr.zip", expected="known cell IDs"
        )
    if file_labels is None:
        # Integer and hexadecimal families: a nucleus carries its cell's label.
        return store.cell_labels[rows]

    # Polygon sets: label = polygon index + 1 of the named set, owned by cell_index.
    labels = _uint32_labels(file_labels, path, context)
    if kind == "cell":
        agrees = np.array_equal(labels, store.cell_labels[rows])
    elif store.nucleus_cells is None:
        raise ReaderFormatError(
            context=context, path=path, found="no nucleus polygon set in cells.zarr.zip", expected="a nucleus set"
        )
    else:
        owners = store.nucleus_cells
        agrees = not (labels > len(owners)).any() and np.array_equal(owners[labels - 1], rows)
    if not agrees:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="label_id values owned by other cells in cells.zarr.zip",
            expected="label_id = polygon index + 1 of the stored set, owned by the polygon's cell",
        )
    return labels


def apply_policy(polygons: Polygons, policy: InvalidGeometryPolicy) -> NDArray[np.bool_]:
    """Return which polygons remain; rings too short to build count as invalid entities.

    Raises
    ------
    ReaderFormatError
        Under ``"raise"`` for any invalid or degenerate polygon, and under
        ``"keep"`` for degenerate rings, which cannot be represented.
    """
    built = ~polygons.degenerate
    # Report the identifiers that index the returned element, only when needed.
    identifiers = polygons.labels if polygons.kind == "nucleus" and polygons.labels is not None else polygons.cell_ids
    if policy == "keep":
        if polygons.degenerate.any():
            examples = identifiers[np.flatnonzero(polygons.degenerate)[:10]].tolist()
            raise ReaderFormatError(
                context=polygons.context,
                path=polygons.path,
                found=f"{int(polygons.degenerate.sum())} rings with fewer than four coordinates, e.g. {examples!r}",
                expected="buildable polygons; invalid_geometries='drop' removes them as invalid entities",
            )
        return built

    invalid = polygons.degenerate.copy()
    invalid[built] = invalid_geometry_mask(polygons.geometries[built])
    if policy == "raise" and invalid.any():
        raise_invalid_geometries(invalid, identifiers.tolist(), context=polygons.context, path=polygons.path)
    return ~invalid


def _parse(frame: gpd.GeoDataFrame, polygons: Polygons, pixel_size: float) -> gpd.GeoDataFrame:
    if frame.empty:
        raise ReaderFormatError(
            context=polygons.context,
            path=polygons.path,
            found="no polygons to return",
            expected=f"at least one polygon; use include_{polygons.kind}_boundaries=False",
        )
    # Register pixel-index vertices in the corner frame without changing them.
    half = pixel_size / 2
    return ShapesModel.parse(frame, transformations={"global": Translation([half, half], axes=("x", "y"))})


def make_cell_polygons(polygons: Polygons, kept: NDArray[np.bool_], pixel_size: float) -> gpd.GeoDataFrame:
    """Index cell polygons by ``cell_id`` in the registered micrometre frame."""
    index = polygons.cell_ids[kept].rename(CELL_ID)
    frame = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygons.geometries[kept], index=index))
    return _parse(frame, polygons, pixel_size)


def make_nucleus_polygons(polygons: Polygons, kept: NDArray[np.bool_], pixel_size: float) -> gpd.GeoDataFrame:
    """Index nucleus polygons by nucleus mask label, with their owning ``cell_id``."""
    if polygons.labels is None:
        raise ReaderFormatError(
            context=polygons.context,
            path=polygons.path,
            found="hexadecimal-ID nuclei without cells.zarr.zip",
            expected="seg_mask_value labels from cells.zarr.zip; use include_nucleus_boundaries=False",
        )
    index = pd.Index(polygons.labels[kept], name=LABEL_ID)
    cell_ids = pd.Series(polygons.cell_ids[kept].array, index=index)
    frame = gpd.GeoDataFrame({CELL_ID: cell_ids}, geometry=gpd.GeoSeries(polygons.geometries[kept], index=index))
    return _parse(frame, polygons, pixel_size)


def make_circles(cells: Cells, kept: NDArray[np.bool_]) -> gpd.GeoDataFrame:
    """Return centroid circles whose area equals the exact mask ``cell_area``."""
    centres = cells.frame[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)[kept]
    radius = np.sqrt(cells.frame["cell_area"].to_numpy(dtype=np.float64)[kept] / np.pi)
    index = cells.ids[kept].rename(CELL_ID)
    geometry = gpd.GeoSeries(shapely.points(centres), index=index)
    frame = gpd.GeoDataFrame({"radius": pd.Series(radius, index=index)}, geometry=geometry)
    return ShapesModel.parse(frame, transformations={"global": Identity()})
