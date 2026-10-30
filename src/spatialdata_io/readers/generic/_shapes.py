"""Generic GeoJSON loading and SpatialData shape-model construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.geometry import (
    geojson_polygons,
    invalid_geometry_mask,
    raise_invalid_geometries,
    validate_invalid_geometry_policy,
)

if TYPE_CHECKING:
    from geopandas import GeoDataFrame

VALID_SHAPE_TYPES = (".geojson",)
_CONTEXT = ReaderErrorContext("generic", "GeoJSON shapes")


def _error(path: Path, found: str) -> ReaderFormatError:
    return ReaderFormatError(
        context=_CONTEXT,
        path=path,
        found=found,
        expected="a nonempty collection of finite 2D Polygon or MultiPolygon geometries",
    )


def _collection(document: object, path: Path) -> list[object]:
    if not isinstance(document, dict):
        raise _error(path, "a non-object document")
    kind = document.get("type")
    if kind == "FeatureCollection":
        features = document.get("features")
        if not isinstance(features, list):
            raise _error(path, "missing or malformed features")
        values: list[object] = []
        for position, feature in enumerate(features):
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                raise _error(path, f"malformed feature {position}")
            values.append(feature.get("geometry"))
    elif kind == "GeometryCollection":
        members = document.get("geometries")
        if not isinstance(members, list):
            raise _error(path, "missing or malformed geometries")
        values = members
    else:
        raise _error(path, f"unsupported top-level type {kind!r}")
    if not values:
        raise _error(path, "an empty collection")
    return values


def geojson(
    path: str | Path, coordinate_system: str, *, invalid_geometries: Literal["keep", "drop", "raise"] = "keep"
) -> GeoDataFrame:
    """Read GeoJSON polygons as a SpatialData shapes element.

    Parameters
    ----------
    path
        GeoJSON file containing polygons or multipolygons.
    coordinate_system
        Source coordinate system receiving an identity transformation. Units
        are those of the input; geographic coordinates are not reprojected.
    invalid_geometries
        Keep invalid topology unchanged, drop entire invalid features, or
        raise. Missing, empty, nonpolygon, nonfinite, nonplanar, open-ring
        and float64-unrepresentable geometries fail under every policy.
        Dropping every feature fails.

    Returns
    -------
    geopandas.GeoDataFrame
        Eager shapes in source order with positional integer IDs. Surviving
        IDs are not renumbered. Feature IDs, properties and CRS declarations
        are discarded, preserving the established output contract.

    Raises
    ------
    ValueError
        Options, document structure or geometry violate the supported contract.
    OSError
        The source cannot be opened or read.

    Notes
    -----
    FeatureCollection and GeometryCollection documents are supported. The
    standard JSON parser decodes the document once; geometries are validated
    and constructed exactly by the shared GeoJSON polygon boundary, and no
    geometry is closed, rounded or repaired. A single geometry
    or Feature document is unsupported. No caller-owned object is mutated.
    """
    policy = validate_invalid_geometry_policy(invalid_geometries)
    source_path = Path(path)
    try:
        with source_path.open(encoding="utf-8") as source:
            document: object = json.load(source)
    except OSError as error:
        add_reader_context(error, context=_CONTEXT, path=source_path)
        raise
    except (UnicodeError, ValueError, RecursionError) as error:
        raise _error(source_path, "malformed JSON") from error
    geometries = geojson_polygons(_collection(document, source_path), context=_CONTEXT, path=source_path)
    frame = gpd.GeoDataFrame(geometry=geometries)
    if policy != "keep":
        invalid = invalid_geometry_mask(frame.geometry.to_numpy())
        if policy == "raise":
            raise_invalid_geometries(invalid, range(len(frame)), context=_CONTEXT, path=source_path)
        elif invalid.any():
            frame = frame.loc[~invalid]
            if frame.empty:
                raise _error(source_path, "all geometries dropped, leaving an empty element")
    return ShapesModel.parse(frame, transformations={coordinate_system: Identity()})
