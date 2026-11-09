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

from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
import shapely
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.csv import csv_errors, csv_header
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.geometry import (
    invalid_geometry_mask,
    raise_invalid_geometries,
    validate_invalid_geometry_policy,
)
from spatialdata_io.readers._utils.paths import require_file
from spatialdata_io.readers.xenium._schemas import SELECTION_TYPES

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.geometry import InvalidGeometryPolicy
    from spatialdata_io.readers._utils.paths import ArtifactFile

_NAME_PREFIX = "#Selection name:"
_MIN_VERTICES = 3
_COMMENT_LINES = 2
_CONTEXT = ReaderErrorContext("xenium", "Explorer selection")
_EXPECTED = "two '#' comment lines, then 'Selection,X,Y' or 'X,Y' records of finite numbers"


def _error(path: Path, *, found: str) -> ReaderFormatError:
    return ReaderFormatError(context=_CONTEXT, path=path, found=found, expected=_EXPECTED)


def _selection_name(header: tuple[str, ...], comment: str, path: ArtifactFile) -> str | None:
    """Resolve the single-selection and named-selection header conventions."""
    if header == ("Selection", "X", "Y"):
        return None
    if header != ("X", "Y"):
        raise _error(path, found=f"header {list(header)!r}")
    name = comment.removeprefix(_NAME_PREFIX).strip()
    if not comment.startswith(_NAME_PREFIX) or not name:
        raise _error(path, found=f"comment {comment!r} without a selection name")
    return name


def _parse(path: ArtifactFile) -> tuple[list[str], NDArray[np.float64]]:
    """Parse one owned stream: two comment lines, the header, then typed records."""
    with path.open("rb") as handle, csv_errors(context=_CONTEXT, path=path):
        comments = [handle.readline().decode("utf-8-sig").rstrip("\r\n") for _ in range(_COMMENT_LINES)]
        if not all(comment.startswith("#") for comment in comments):
            raise _error(path, found="records without two leading comment lines")
        header = csv_header(path, context=_CONTEXT, source=handle)
        single_name = _selection_name(header, comments[0], path)
        # Arrow reads the header again, so a header-only file is an empty table.
        handle.seek(0)
        table = pacsv.read_csv(
            handle,
            read_options=pacsv.ReadOptions(skip_rows=_COMMENT_LINES),
            convert_options=pacsv.ConvertOptions(
                column_types=SELECTION_TYPES, include_columns=list(header), null_values=[], strings_can_be_null=False
            ),
        )
    if not table.num_rows:
        raise _error(path, found="a header without records")
    names = [single_name] * table.num_rows if single_name is not None else table["Selection"].to_pylist()
    vertices = np.column_stack([table["X"].to_numpy(), table["Y"].to_numpy()])
    if "" in names or not np.isfinite(vertices).all():
        raise _error(path, found="unnamed selections or nonfinite coordinates")
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
    names, coordinates = _parse(file)

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
