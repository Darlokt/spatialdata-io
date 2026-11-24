"""Read selection polygons exported by Xenium Explorer.

CSV coordinates are in micrometres in the morphology image frame. GeoJSON
coordinates are pixels in the morphology or post-instrument image chosen
at export.
https://www.10xgenomics.com/support/software/xenium-explorer/latest/tutorials/xe-selecting-multiple-regions-of-interest
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import shapely
from spatialdata.models import ShapesModel
from spatialdata.transformations import Affine, BaseTransformation, Identity

from spatialdata_io.readers._utils.csv import csv_errors, csv_header
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.geometry import (
    geojson_polygons,
    invalid_geometry_mask,
    raise_invalid_geometries,
    validate_invalid_geometry_policy,
)
from spatialdata_io.readers._utils.metadata import read_json_object
from spatialdata_io.readers._utils.paths import require_file
from spatialdata_io.readers.xenium._schemas import SELECTION_TYPES, STRING_DTYPE

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.geometry import InvalidGeometryPolicy
    from spatialdata_io.readers._utils.paths import ArtifactFile

_NAME_PREFIX = "#Selection name:"
_MIN_VERTICES = 3
_COMMENT_LINES = 2
_CONTEXT = ReaderErrorContext("xenium", "Explorer selection")
_ANNOTATED_CSV = ("Selection", "X", "Y", "Class", "Color")
_EXPECTED_CSV = "two '#' comment lines, then 'Selection,X,Y,Class,Color', 'Selection,X,Y' or 'X,Y' records"
_EXPECTED_GEOJSON = "a Xenium Explorer FeatureCollection of named, colored Polygon annotations with finite XY rings"
_XY_AXES = ("x", "y")
_RGB_CHANNELS = 3
_MAX_COLOR_CHANNEL = 255


def _csv_error(path: Path, *, found: str) -> ReaderFormatError:
    return ReaderFormatError(context=_CONTEXT, path=path, found=found, expected=_EXPECTED_CSV)


def _selection_name(header: tuple[str, ...], comment: str, path: ArtifactFile) -> str | None:
    """Resolve the single-selection and named-selection header conventions."""
    if header in (("Selection", "X", "Y"), _ANNOTATED_CSV):
        return None
    if header != ("X", "Y"):
        raise _csv_error(path, found=f"header {list(header)!r}")
    name = comment.removeprefix(_NAME_PREFIX).strip()
    if not comment.startswith(_NAME_PREFIX) or not name:
        raise _csv_error(path, found=f"comment {comment!r} without a selection name")
    return name


def _csv_annotations(frame: gpd.GeoDataFrame, table: pa.Table, starts: list[int], path: ArtifactFile) -> None:
    """Keep the constant class and color fields of each selection."""
    rows = pa.array(starts, type=pa.int64())
    ends = [*starts[1:], table.num_rows]
    for source, column in (("Class", "annotation_class"), ("Color", "color")):
        values = table[source]
        selected = values.take(rows)
        for row, (start, end, value) in enumerate(zip(starts, ends, selected, strict=True)):
            equal = pc.call_function("equal", [values.slice(start, end - start), value])
            if not pc.call_function("all", [equal]).as_py():
                raise _csv_error(path, found=f"selection {frame.index[row]!r} with inconsistent {source}")
        frame[column] = pd.Series(selected, index=frame.index, dtype=STRING_DTYPE)


def _read_csv(path: ArtifactFile) -> gpd.GeoDataFrame:
    """Read typed records, grouping contiguous vertices and annotations by name."""
    with path.open("rb") as handle, csv_errors(context=_CONTEXT, path=path):
        comments = [handle.readline().decode("utf-8-sig").rstrip("\r\n") for _ in range(_COMMENT_LINES)]
        if not all(comment.startswith("#") for comment in comments):
            raise _csv_error(path, found="records without two leading comment lines")
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
        raise _csv_error(path, found="a header without records")
    names = [single_name] * table.num_rows if single_name is not None else table["Selection"].to_pylist()
    vertices = np.column_stack([table["X"].to_numpy(), table["Y"].to_numpy()])
    if "" in names or not np.isfinite(vertices).all():
        raise _csv_error(path, found="unnamed selections or nonfinite coordinates")

    order = list(dict.fromkeys(names))
    starts = [index for index, name in enumerate(names) if index == 0 or names[index - 1] != name]
    if len(starts) != len(order):
        raise ReaderFormatError(
            context=_CONTEXT,
            path=path,
            found="a selection name in several runs",
            expected="contiguous rows per selection",
        )
    ends = [*starts[1:], len(names)]
    polygons = []
    for name, start, end in zip(order, starts, ends, strict=True):
        ring = vertices[start:end]
        if len(np.unique(ring, axis=0)) < _MIN_VERTICES:
            raise ReaderFormatError(
                context=_CONTEXT,
                path=path,
                found=f"selection {name!r} with fewer than three distinct vertices",
                expected="polygonal selections",
            )
        polygons.append(shapely.Polygon(ring))
    index = pd.Index(order, dtype="str", name="selection")
    frame = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygons, index=index))
    if header == _ANNOTATED_CSV:
        _csv_annotations(frame, table, starts, path)
    return frame


def _geojson_object(value: object, *, member: str, path: ArtifactFile) -> dict[str, object]:
    if type(value) is not dict:
        raise ReaderFormatError(
            context=_CONTEXT, path=path, found=f"{member} without an object", expected=_EXPECTED_GEOJSON
        )
    # read_json_object decodes all objects as dictionaries with text keys.
    return cast("dict[str, object]", value)


def _hex_color(color: object, *, member: str, path: ArtifactFile) -> str:
    """Represent an RGB triplet without losing any of its channel values."""
    if (
        type(color) is not list
        or len(color) != _RGB_CHANNELS
        or any(type(value) is not int or not 0 <= value <= _MAX_COLOR_CHANNEL for value in color)
    ):
        raise ReaderFormatError(
            context=_CONTEXT, path=path, found=f"invalid {member} RGB color", expected=_EXPECTED_GEOJSON
        )
    return "#" + "".join(f"{value:02x}" for value in color)


def _z_plane(geometry: dict[str, object], *, feature: int, path: ArtifactFile) -> int | None:
    if "plane" not in geometry:
        return None
    plane = _geojson_object(geometry["plane"], member=f"feature {feature} plane", path=path)
    z = plane.get("z")
    if plane.keys() != {"z"} or type(z) is not int or not 0 <= z < 2**32:
        raise ReaderFormatError(
            context=_CONTEXT,
            path=path,
            found=f"feature {feature} with unsupported geometry plane {plane!r}",
            expected="a geometry plane containing one uint32 z value",
        )
    return z


def _annotation_class(
    properties: dict[str, object], *, feature: int, path: ArtifactFile
) -> tuple[str | None, str | None]:
    if "classification" not in properties:
        return None, None
    label = _geojson_object(properties["classification"], member=f"feature {feature} classification", path=path)
    name = label.get("name")
    if not isinstance(name, str) or not name:
        raise ReaderFormatError(
            context=_CONTEXT, path=path, found=f"feature {feature} without a class name", expected=_EXPECTED_GEOJSON
        )
    return name, _hex_color(label.get("color"), member=f"feature {feature} class", path=path)


def _read_geojson(path: ArtifactFile) -> gpd.GeoDataFrame:
    """Read Explorer's exported polygon and top-level annotation properties."""
    document = read_json_object(path, context=_CONTEXT)
    features = document.get("features")
    if document.get("type") != "FeatureCollection" or type(features) is not list or not features:
        raise ReaderFormatError(
            context=_CONTEXT, path=path, found="no nonempty FeatureCollection", expected=_EXPECTED_GEOJSON
        )
    names: list[str] = []
    colors: list[str] = []
    classes: list[str | None] = []
    class_colors: list[str | None] = []
    planes: list[int | None] = []
    geometries: list[dict[str, object]] = []
    for index, item in enumerate(features):
        feature = _geojson_object(item, member=f"feature {index}", path=path)
        geometry = _geojson_object(feature.get("geometry"), member=f"feature {index} geometry", path=path)
        properties = _geojson_object(feature.get("properties"), member=f"feature {index} properties", path=path)
        name = properties.get("name")
        if (
            feature.get("type") != "Feature"
            or geometry.get("type") != "Polygon"
            or properties.get("objectType") != "annotation"
            or not isinstance(name, str)
            or not name
        ):
            raise ReaderFormatError(
                context=_CONTEXT, path=path, found=f"malformed feature {index}", expected=_EXPECTED_GEOJSON
            )
        names.append(name)
        colors.append(_hex_color(properties.get("color"), member=f"feature {index}", path=path))
        coordinates = geometry.get("coordinates")
        if type(coordinates) is list and len(coordinates) > 1:
            raise ReaderFormatError(
                context=_CONTEXT,
                path=path,
                found=f"selection {name!r} with interior rings",
                expected="one closed 2D ring per Explorer annotation",
            )
        geometries.append(geometry)
        planes.append(_z_plane(geometry, feature=index, path=path))
        class_name, class_color = _annotation_class(properties, feature=index, path=path)
        classes.append(class_name)
        class_colors.append(class_color)
    try:
        index = pd.Index(names, dtype="str", name="selection")
        columns = {"color": pd.Series(colors, index=index, dtype=STRING_DTYPE)}
        if any(name is not None for name in classes):
            columns["annotation_class"] = pd.Series(classes, index=index, dtype=STRING_DTYPE)
            columns["annotation_class_color"] = pd.Series(class_colors, index=index, dtype=STRING_DTYPE)
    except UnicodeEncodeError as error:
        raise ReaderFormatError(
            context=_CONTEXT, path=path, found="an invalid Unicode name", expected="UTF-8 selection and class names"
        ) from error
    if not index.is_unique:
        duplicate = index[index.duplicated()][0]
        raise ReaderFormatError(
            context=_CONTEXT,
            path=path,
            found=f"duplicate selection name {duplicate!r}",
            expected="unique selection names",
        )
    if any(plane is not None for plane in planes):
        columns["z_plane"] = pd.Series(planes, index=index, dtype=pd.ArrowDtype(pa.uint32()))
    polygons = geojson_polygons(geometries, context=_CONTEXT, path=path)
    return gpd.GeoDataFrame(columns, geometry=gpd.GeoSeries(polygons, index=index))


def _image_registration(transformation: BaseTransformation | None) -> Affine:
    if transformation is None:
        message = (
            "image_to_global is required for Explorer GeoJSON. Use the global transform of the image chosen at export."
        )
        raise ValueError(message)
    if not isinstance(transformation, BaseTransformation):
        message = "image_to_global must be a SpatialData transformation."
        raise TypeError(message)
    matrix = np.asarray(transformation.to_affine_matrix(input_axes=_XY_AXES, output_axes=_XY_AXES), dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or not np.array_equal(matrix[2], [0.0, 0.0, 1.0])
        or np.linalg.slogdet(matrix[:2, :2])[0] == 0
    ):
        message = "image_to_global must be a finite, invertible XY affine transform."
        raise ValueError(message)
    return Affine(matrix, input_axes=_XY_AXES, output_axes=_XY_AXES)


def xenium_explorer_selection(
    path: str | Path,
    *,
    image_to_global: BaseTransformation | None = None,
    invalid_geometries: InvalidGeometryPolicy = "keep",
) -> gpd.GeoDataFrame:
    """Read Xenium Explorer CSV or GeoJSON selections as registered polygons.

    Parameters
    ----------
    path
        Coordinates CSV or ``.geojson`` file exported by Xenium Explorer.
    image_to_global
        Required for GeoJSON. The transformation from full-resolution XY
        pixels of the morphology or post-instrument image chosen at export to
        the Xenium ``global`` coordinate system in micrometres. Use the transform
        of the matching image already read by :func:`spatialdata_io.xenium`
        or :func:`spatialdata_io.xenium_aligned_image`. Omit for CSV.
    invalid_geometries
        ``"keep"`` returns hand-drawn polygons as exported even when they
        self-intersect; ``"drop"`` removes invalid selections; ``"raise"``
        rejects them. Validity means ``shapely.is_valid``.

    Returns
    -------
    geopandas.GeoDataFrame
        Eager polygons indexed by selection name in file order. CSV vertices
        are in micrometres with ``Identity()`` into ``global``. GeoJSON vertices
        stay in source image pixels with ``image_to_global`` into ``global``.
        GeoJSON colors are stored as hexadecimal RGB strings in ``color``.
        Classified GeoJSON annotations also have ``annotation_class`` and
        ``annotation_class_color``. A GeoJSON plane tag is kept in ``z_plane``
        as an integer annotation on the 2D geometry.
        CSVs with ``Class`` and ``Color`` add ``annotation_class`` and ``color``.

    Raises
    ------
    TypeError, ValueError
        If ``invalid_geometries`` or ``image_to_global`` is invalid.
    FileNotFoundError, IsADirectoryError
        If the file is missing or a directory.
    ValueError
        A contextual ``ReaderFormatError`` if the file is malformed, a
        CSV selection's rows are not contiguous or have fewer than three
        distinct vertices, invalid geometry is rejected, or no selection remains.
    """
    policy = validate_invalid_geometry_policy(invalid_geometries)
    source = Path(path).expanduser()
    geojson = source.suffix.lower() == ".geojson"
    if not geojson and image_to_global is not None:
        message = "image_to_global is only supported for GeoJSON. Explorer CSV coordinates are already in micrometres."
        raise ValueError(message)
    file = require_file(source, context=_CONTEXT)
    transformation = _image_registration(image_to_global) if geojson else Identity()
    frame = _read_geojson(file) if geojson else _read_csv(file)
    if policy != "keep":
        invalid = invalid_geometry_mask(frame.geometry.to_numpy())
        if policy == "raise":
            raise_invalid_geometries(invalid, frame.index, context=_CONTEXT, path=file)
        elif invalid.any():
            kept = ~invalid
            if not kept.any():
                raise ReaderFormatError(
                    context=_CONTEXT, path=file, found="no valid selection", expected="at least one valid selection"
                )
            frame = frame.iloc[kept]
    return ShapesModel.parse(frame, transformations={"global": transformation})
