"""Align linked instance identities and explicitly derive target centres."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from spatialdata import SpatialData, get_centroids
from spatialdata.models import get_axes_names, get_table_keys
from spatialdata.transformations import get_transformation

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray, DataTree

from spatialdata_io.converters.legacy_anndata._schemas import physical_coordinates


@dataclass(frozen=True, slots=True)
class LinkedTarget:
    """Associate selected table rows with one existing target element."""

    name: str
    rows: NDArray[np.int64]
    instances: pd.Index
    element: GeoDataFrame | DaskDataFrame | DataArray | DataTree


def _validate_shape_ids(element: GeoDataFrame, instances: pd.Index, *, name: str) -> None:
    if element.index.nlevels != 1 or element.index.hasnans or not element.index.is_unique:
        message = f"Export target {name!r} requires unique non-null instance identifiers."
        raise ValueError(message)
    if (element.index.get_indexer(instances) < 0).any():
        message = f"Export target {name!r} is missing referenced instance identifiers."
        raise ValueError(message)


def _linked_identifiers(table: AnnData) -> tuple[pd.Series, pd.Series]:
    try:
        regions, region_key, instance_key = get_table_keys(table)
    except (KeyError, TypeError) as error:
        message = "Export requires valid spatialdata_attrs with region, region_key and instance_key."
        raise ValueError(message) from error
    if (
        not isinstance(region_key, str)
        or not isinstance(instance_key, str)
        or not region_key
        or not instance_key
        or region_key == instance_key
        or region_key not in table.obs
        or instance_key not in table.obs
        or table.obs.columns.tolist().count(region_key) != 1
        or table.obs.columns.tolist().count(instance_key) != 1
    ):
        message = "Export requires distinct existing region and instance observation columns in spatialdata_attrs."
        raise ValueError(message)
    declared = [regions] if isinstance(regions, str) else regions
    if isinstance(declared, np.ndarray) and declared.ndim == 1:
        declared = declared.tolist()
    if not isinstance(declared, list) or any(not isinstance(name, str) or not name for name in declared):
        message = "Export requires nonempty string target names in spatialdata_attrs."
        raise ValueError(message)
    # Conversion validates eager pandas annotations before this boundary.
    observations = cast("pd.DataFrame", table.obs)
    region_values = observations[region_key]
    instances = observations[instance_key]
    if (
        region_values.isna().any()
        or instances.isna().any()
        or not region_values.isin(declared).all()
        or pd.api.types.infer_dtype(instances, skipna=False) not in {"integer", "string", "unicode", "empty"}
    ):
        message = "Export requires declared non-null regions and homogeneous integer or string instance identifiers."
        raise ValueError(message)
    if not pd.MultiIndex.from_arrays([region_values, instances]).is_unique:
        message = "Export requires unique (region, instance) pairs."
        raise ValueError(message)
    return region_values, instances


def linked_targets(sdata: SpatialData, table: AnnData) -> tuple[LinkedTarget, ...]:
    """Validate identities without evaluating lazy target membership."""
    region_values, instances = _linked_identifiers(table)
    targets: list[LinkedTarget] = []
    for name in pd.unique(region_values):
        if not isinstance(name, str) or (
            name not in sdata.shapes and name not in sdata.labels and name not in sdata.points
        ):
            message = f"Export target {name!r} is not an available Shapes, Labels or Points element."
            raise ValueError(message)
        rows = np.flatnonzero(region_values.eq(name).to_numpy(dtype=bool))
        ids = pd.Index(instances.iloc[rows])
        if name in sdata.shapes:
            element = sdata.shapes[name]
            _validate_shape_ids(element, ids, name=name)
            targets.append(LinkedTarget(name, rows, ids, element))
        elif name in sdata.labels:
            targets.append(LinkedTarget(name, rows, ids, sdata.labels[name]))
        else:
            targets.append(LinkedTarget(name, rows, ids, sdata.points[name]))
    return tuple(targets)


def _point_coordinates(target: LinkedTarget, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    # The upstream Point centroid path transforms before checking integer
    # precision. Validate the projected source first, in its only source scan.
    points = cast("DaskDataFrame", target.element)[["x", "y"]].compute()
    if points.index.hasnans or not points.index.is_unique:
        message = f"Export target {target.name!r} requires unique non-null instance identifiers."
        raise ValueError(message)
    locations = points.index.get_indexer(target.instances)
    if (locations < 0).any():
        message = f"Export target {target.name!r} is missing referenced instance identifiers."
        raise ValueError(message)
    selected = points.iloc[locations]
    coordinates = np.empty((len(locations), 2), dtype=np.float64)
    for axis, key in enumerate(("x", "y")):
        values = selected[key].to_numpy()
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            message = f"Export target {target.name!r} requires finite real numerical {key!r} coordinates."
            raise ValueError(message)
        coordinates[:, axis] = physical_coordinates(values, key=f"{target.name}.{key}")
    transformed = coordinates @ matrix[:2, :2].T
    transformed += matrix[:2, 2]
    return transformed


def _target_affine(target: LinkedTarget, frame: str) -> NDArray[np.float64]:
    axes = get_axes_names(target.element)
    if set(axes) != {"x", "y"}:
        message = f"Export target {target.name!r} requires XY axes; found {axes!r}."
        raise ValueError(message)
    # get_all=True is documented to return a mapping; upstream annotations
    # do not discriminate it from the single-transformation return type.
    transformations = cast("dict[str, BaseTransformation]", get_transformation(target.element, get_all=True))
    if frame not in transformations:
        message = f"Export target {target.name!r} has no transformation to frame {frame!r}."
        raise ValueError(message)
    matrix = transformations[frame].to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    if not np.isfinite(matrix).all() or np.linalg.slogdet(matrix[:2, :2])[0] == 0:
        message = f"Export target {target.name!r} requires a finite invertible XY transformation to {frame!r}."
        raise ValueError(message)
    # SpatialData constructs float64 homogeneous matrices; its return
    # annotation retains the broader floating scalar family.
    return cast("NDArray[np.float64]", matrix)


def _geometry_coordinates(target: LinkedTarget, frame: str) -> NDArray[np.float64]:
    try:
        # Only referenced eager geometries need centroid calculation.
        element = target.element.loc[target.instances] if isinstance(target.element, GeoDataFrame) else target.element
        centres = get_centroids(element, coordinate_system=frame).compute()
    except (ValueError, TypeError) as error:
        error.add_note(f"while deriving centres for export target {target.name!r} in frame {frame!r}")
        raise
    if centres.index.hasnans or not centres.index.is_unique:
        message = f"Centres for export target {target.name!r} require unique non-null instance identifiers."
        raise ValueError(message)
    locations = centres.index.get_indexer(target.instances)
    if (locations < 0).any():
        message = f"Export target {target.name!r} is missing referenced instance identifiers."
        raise ValueError(message)
    # The public centroid operation retains instance IDs, not table ordering.
    return centres.iloc[locations].loc[:, ["x", "y"]].to_numpy(dtype=np.float64)


def centroid_coordinates(targets: tuple[LinkedTarget, ...], *, frame: str, n_obs: int) -> NDArray[np.float64]:
    """Derive centres in one established XY frame, restoring table row order."""
    # Validate every selected mapping before any lazy source is evaluated.
    prepared = [(target, _target_affine(target, frame)) for target in targets]
    coordinates = np.empty((n_obs, 2), dtype=np.float64)
    for target, matrix in prepared:
        coordinates[target.rows] = (
            _point_coordinates(target, matrix)
            if isinstance(target.element, DaskDataFrame)
            else _geometry_coordinates(target, frame)
        )
    if not np.isfinite(coordinates).all():
        message = f"Derived export centres in frame {frame!r} must be finite float64 XY positions."
        raise ValueError(message)
    return coordinates
