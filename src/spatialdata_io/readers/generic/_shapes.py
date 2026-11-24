"""Read one caller-selected GeoJSON file as SpatialData polygon shapes.

Two document kinds of RFC 7946 (https://www.rfc-editor.org/rfc/rfc7946) are
supported; every geometry must be a 2D ``Polygon`` or ``MultiPolygon``::

    {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": ..., "properties": {...}, "geometry": {"type": "Polygon", "coordinates": [...]}},
            ...,
        ],
    }

    {"type": "GeometryCollection", "geometries": [{"type": "MultiPolygon", "coordinates": [...]}, ...]}

The document is strict UTF-8 JSON: duplicate object names, ``NaN`` and
``Infinity`` are rejected rather than resolved by the parser. Rows keep source
order and use positional integer IDs by default. Callers can select top-level
Feature IDs instead. Properties and any ``crs`` member are discarded.
Coordinates keep the units of the source (pixels, micrometres or geographic
degrees) and become float64. Rings are never reprojected, closed or repaired.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import pandas as pd
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.geometry import (
    geojson_polygons,
    invalid_geometry_mask,
    raise_invalid_geometries,
    validate_invalid_geometry_policy,
)
from spatialdata_io.readers._utils.metadata import read_json_object
from spatialdata_io.readers._utils.paths import require_file

if TYPE_CHECKING:
    from geopandas import GeoDataFrame

VALID_SHAPE_TYPES = (".geojson",)
_CONTEXT = ReaderErrorContext("generic", "GeoJSON shapes")
_FEATURE_IDS = "a unique nonempty string or signed 64-bit integer 'id' on every Feature, all of the same type"


def _error(path: Path, found: str) -> ReaderFormatError:
    return ReaderFormatError(
        context=_CONTEXT,
        path=path,
        found=found,
        expected="a nonempty collection of finite 2D Polygon or MultiPolygon geometries",
    )


def _feature_index(identifiers: list[object], path: Path) -> pd.Index:
    """Validate all source IDs before pandas can coerce them."""
    identifier_type = type(identifiers[0])
    for position, identifier in enumerate(identifiers):
        if type(identifier) is not identifier_type or not (
            (isinstance(identifier, str) and identifier) or (type(identifier) is int and -(2**63) <= identifier < 2**63)
        ):
            raise ReaderFormatError(
                context=_CONTEXT,
                path=path,
                found=f"Feature id {identifier!r} at position {position}",
                expected=_FEATURE_IDS,
            )
    dtype = pd.StringDtype(storage="pyarrow", na_value=float("nan")) if identifier_type is str else "int64"
    try:
        index = pd.Index(identifiers, dtype=dtype)
    except UnicodeEncodeError as error:
        raise ReaderFormatError(
            context=_CONTEXT, path=path, found="a Feature id that cannot be encoded as UTF-8", expected=_FEATURE_IDS
        ) from error
    if not index.is_unique:
        duplicate = identifiers[int(index.duplicated().argmax())]
        raise ReaderFormatError(
            context=_CONTEXT, path=path, found=f"duplicate Feature id {duplicate!r}", expected=_FEATURE_IDS
        )
    return index


def _geojson_rows(document: dict[str, object], path: Path, *, use_feature_ids: bool) -> tuple[list[object], pd.Index]:
    """Return geometries and their index in source order."""
    kind = document.get("type")
    identifiers: list[object] = []
    if kind == "FeatureCollection":
        features = document.get("features")
        if not isinstance(features, list):
            raise _error(path, "missing or malformed features")
        geometries: list[object] = []
        for position, feature in enumerate(features):
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                raise _error(path, f"malformed feature {position}")
            geometries.append(feature.get("geometry"))
            if use_feature_ids:
                identifiers.append(feature.get("id"))
    elif kind == "GeometryCollection":
        if use_feature_ids:
            raise ReaderFormatError(
                context=_CONTEXT,
                path=path,
                found="a GeometryCollection",
                expected="a FeatureCollection with Feature IDs",
            )
        members = document.get("geometries")
        if not isinstance(members, list):
            raise _error(path, "missing or malformed geometries")
        geometries = members
    else:
        raise _error(path, f"unsupported top-level type {kind!r}")
    if not geometries:
        raise _error(path, "an empty collection")
    index = _feature_index(identifiers, path) if use_feature_ids else pd.RangeIndex(len(geometries))
    return geometries, index


def geojson(
    path: str | Path,
    coordinate_system: str,
    *,
    use_feature_ids: bool = False,
    invalid_geometries: Literal["keep", "drop", "raise"] = "keep",
) -> GeoDataFrame:
    """Read GeoJSON polygons as a SpatialData shapes element.

    ``FeatureCollection`` and ``GeometryCollection`` documents of 2D
    ``Polygon`` and ``MultiPolygon`` geometries are supported; a single
    geometry or ``Feature`` document is not.

    Parameters
    ----------
    path
        GeoJSON file. ``~`` is expanded.
    coordinate_system
        Name of the coordinate system of the source coordinates, preserved
        literally; it must contain a non-whitespace character.
    use_feature_ids
        Use top-level ``id`` values from a ``FeatureCollection`` as the index.
        Every feature must have a unique ID, and IDs must all be nonempty UTF-8
        strings or all be JSON integers in the signed 64-bit range. Text is preserved
        literally and integers are stored as ``int64``. IDs are validated before
        any invalid geometry is dropped. The default uses positional IDs.
    invalid_geometries
        Policy for polygons that are not valid by ``shapely.is_valid``, such
        as self-intersecting rings: ``"keep"`` preserves them unchanged,
        ``"drop"`` removes their entire rows without renumbering the others,
        and ``"raise"`` fails. Dropping every row fails.

    Returns
    -------
    geopandas.GeoDataFrame
        Eager shapes with a single ``geometry`` column, one row per feature or
        geometry in source order, and the selected index.

        - Vertices become float64; holes and multipart order follow the source.
          Nothing is closed or repaired.
        - Feature properties and ``crs`` declarations are discarded.
        - Coordinates: an ``Identity`` to ``coordinate_system``; the units are
          those of the source and geographic coordinates are not reprojected.

    Raises
    ------
    TypeError
        If ``coordinate_system`` or ``invalid_geometries`` is not a string,
        or ``use_feature_ids`` is not a boolean.
    ValueError
        If ``coordinate_system`` or ``invalid_geometries`` has an invalid value.
    ReaderFormatError
        Under every policy, if the file is not strict UTF-8 JSON, the document
        is not a nonempty supported collection, or a geometry is missing,
        empty, nonpolygonal, nonplanar or nonfinite, or has an open or short
        ring.
        Also if ``use_feature_ids=True`` and a collection or its IDs violate
        the Feature ID contract.
        Also for invalid geometries under ``"raise"`` and when ``"drop"``
        removes every row.
    FileNotFoundError, IsADirectoryError
        If ``path`` does not name an existing regular file.
    """
    if not isinstance(coordinate_system, str):
        message = "coordinate_system must be a string."
        raise TypeError(message)
    if not coordinate_system.strip():
        message = "coordinate_system must contain a non-whitespace character."
        raise ValueError(message)
    if not isinstance(use_feature_ids, bool):
        message = "use_feature_ids must be a boolean."
        raise TypeError(message)
    policy = validate_invalid_geometry_policy(invalid_geometries)

    file = require_file(Path(path).expanduser(), context=_CONTEXT)
    document = read_json_object(file, context=_CONTEXT)
    rows, index = _geojson_rows(document, file, use_feature_ids=use_feature_ids)
    geometries = geojson_polygons(rows, context=_CONTEXT, path=file)
    frame = gpd.GeoDataFrame(geometry=geometries, index=index)

    if policy != "keep":
        invalid = invalid_geometry_mask(geometries)
        if policy == "raise":
            raise_invalid_geometries(invalid, index, context=_CONTEXT, path=file)
        elif invalid.any():
            frame = frame.loc[~invalid]
            if frame.empty:
                raise _error(file, "all geometries dropped, leaving an empty element")
    return ShapesModel.parse(frame, transformations={coordinate_system: Identity()})
