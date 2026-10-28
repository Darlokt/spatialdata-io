"""Read Xenium Explorer selection coordinate CSV exports as micrometre polygons.

Every export starts with two ``#`` comment lines. Multi-selection files and
single-selection files from Explorer 4.0 then have the header ``Selection,X,Y``;
older single-selection files have ``X,Y`` and name the selection in their first
comment line (``#Selection name: Selection 1``). Coordinates are micrometres in
the morphology image frame. Clicked cells are exported as ``Cell <id>``
selections.
https://www.10xgenomics.com/support/software/xenium-explorer/latest/tutorials/xe-selecting-multiple-regions-of-interest
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.geometry import (
    invalid_geometry_mask,
    raise_invalid_geometries,
    validate_invalid_geometry_policy,
)
from spatialdata_io.readers._utils.paths import require_file

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.geometry import InvalidGeometryPolicy
    from spatialdata_io.readers._utils.paths import ArtifactFile

MAX_SELECTION_BYTES = 256 * 1024 * 1024
_NAME_PREFIX = "#Selection name:"
_MIN_VERTICES = 3
_COMMENT_LINES = 2
_COORDINATES = 2
_CONTEXT = ReaderErrorContext("xenium", "Explorer selection")
_EXPECTED = "two '#' comment lines, then 'Selection,X,Y' or 'X,Y' records of finite numbers"


def _error(path: Path, *, found: str) -> ReaderFormatError:
    return ReaderFormatError(context=_CONTEXT, path=path, found=found, expected=_EXPECTED)


def _rows(path: ArtifactFile) -> list[list[str]]:
    """Return the nonblank records of a bounded UTF-8 file after its two comment lines are checked."""
    with path.open("rb") as handle:
        payload = handle.read(MAX_SELECTION_BYTES + 1)
    if len(payload) > MAX_SELECTION_BYTES:
        raise _error(path, found="a file larger than 256 MiB")
    try:
        rows = [row for row in csv.reader(payload.decode("utf-8-sig").splitlines(), strict=True) if row]
    except (UnicodeDecodeError, csv.Error) as caught:
        raise _error(path, found="malformed CSV text") from caught
    if len(rows) <= _COMMENT_LINES or not all(row[0].startswith("#") for row in rows[:_COMMENT_LINES]):
        raise _error(path, found=f"{len(rows)} records without two leading comment lines")
    return rows


def _parse(path: ArtifactFile) -> tuple[list[str], list[tuple[float, float]]]:
    """Return one name per vertex row and the vertices in file order."""
    rows = _rows(path)
    header, records = rows[_COMMENT_LINES], rows[_COMMENT_LINES + 1 :]
    if not records:
        raise _error(path, found="a header without records")
    if header == ["X", "Y"]:
        comment = ",".join(rows[0])
        if not comment.startswith(_NAME_PREFIX) or not comment.removeprefix(_NAME_PREFIX).strip():
            raise _error(path, found=f"comment {comment!r} without a selection name")
        names = [comment.removeprefix(_NAME_PREFIX).strip()] * len(records)
    elif header == ["Selection", "X", "Y"]:
        names = [record[0] for record in records]
        records = [record[1:] for record in records]
    else:
        raise _error(path, found=f"header {header!r}")

    try:
        vertices = [(float(record[0]), float(record[1])) for record in records if len(record) == _COORDINATES]
    except ValueError as caught:
        raise _error(path, found="non-numeric coordinates") from caught
    if (
        len(vertices) != len(records)
        or not all(math.isfinite(x) and math.isfinite(y) for x, y in vertices)
        or any(not name for name in names)
    ):
        raise _error(path, found="records of the wrong width, unnamed selections or nonfinite coordinates")
    return names, vertices


def xenium_explorer_selection(
    path: str | Path, *, invalid_geometries: InvalidGeometryPolicy = "keep"
) -> gpd.GeoDataFrame:
    """Read a Xenium Explorer selection CSV as polygons in micrometres.

    Parameters
    ----------
    path
        Coordinates CSV exported from a Xenium Explorer selection or from the
        Selections window (Lasso, Freehand or Rectangular selections and
        clicked cells).
    invalid_geometries
        ``"keep"`` returns hand-drawn polygons as exported even when they
        self-intersect; ``"drop"`` removes invalid selections; ``"raise"``
        rejects them. Validity means ``shapely.is_valid``.

    Returns
    -------
    geopandas.GeoDataFrame
        A ``ShapesModel`` element with one polygon per selection in file order,
        indexed by selection name (``Selection 1``, ``Cell aaaaficg-1``, ...), with
        ``{"global": Identity()}`` in micrometres. It can be added to a
        ``SpatialData`` object read by :func:`spatialdata_io.xenium` or used with
        :func:`spatialdata.polygon_query`.

    Raises
    ------
    TypeError, ValueError
        If ``invalid_geometries`` is invalid.
    FileNotFoundError, IsADirectoryError
        If the file is missing or a directory.
    ValueError
        A contextual ``ReaderFormatError`` if the file is malformed, a
        selection's rows are not contiguous, a selection has fewer than three
        distinct vertices, invalid geometry is rejected, or no selection remains.
    """
    policy = validate_invalid_geometry_policy(invalid_geometries)
    file = require_file(Path(path).expanduser(), context=_CONTEXT)
    names, vertices = _parse(file)

    order = list(dict.fromkeys(names))
    starts = [index for index, name in enumerate(names) if index == 0 or names[index - 1] != name]
    if len(starts) != len(order):
        raise ReaderFormatError(
            context=_CONTEXT,
            path=file,
            found="a selection name in several runs",
            expected="contiguous rows per selection",
        )
    ends = [*starts[1:], len(names)]
    coordinates = np.asarray(vertices, dtype=np.float64)
    polygons = []
    for name, start, end in zip(order, starts, ends, strict=True):
        ring = coordinates[start:end]
        if len(np.unique(ring, axis=0)) < _MIN_VERTICES:
            raise ReaderFormatError(
                context=_CONTEXT,
                path=file,
                found=f"selection {name!r} with fewer than three distinct vertices",
                expected="polygonal selections",
            )
        polygons.append(shapely.Polygon(ring))
    geometries = np.asarray(polygons, dtype=object)

    invalid = invalid_geometry_mask(geometries)
    if policy == "raise":
        raise_invalid_geometries(invalid, order, context=_CONTEXT, path=file)
    kept = ~invalid if policy == "drop" else np.ones(len(order), dtype=bool)
    if not kept.any():
        raise ReaderFormatError(
            context=_CONTEXT, path=file, found="no valid selection", expected="at least one valid selection"
        )
    index = pd.Index(order, dtype="str", name="selection")[kept]
    frame = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geometries[kept], index=index))
    return ShapesModel.parse(frame, transformations={"global": Identity()})
