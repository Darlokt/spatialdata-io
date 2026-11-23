"""Read the cell table (``table_file``) into the linked table and add its identifiers and positions.

The table is read eagerly with AnnData, validated against the schema in
``_schemas`` and then changed in place: the reader owns the freshly read
object, so expression, layers, ``raw`` and annotations are not copied.

- ``obs["cell_id"]`` is the uint64 mask label parsed from the observation name.
  ``obs["region"]`` is the categorical ``<sample>_labels`` of ``obs["image"]``.
- ``obsm["spatial_pixels"]`` holds the regionprops centroids as XY
  (``centroid-1``, ``centroid-0``) in their source dtype. Regionprops averages
  zero-based pixel indices, so these are pixel-index positions: the centre of
  pixel ``(0, 0)`` is ``0``, while the raster frame puts it at ``0.5``.
- When every sample is calibrated, ``obsm["spatial"]`` is float64 micrometres
  ``(xy + 0.5) * spacing_xy_um`` in the sample's raster frame. With one sample
  the frame is fixed, otherwise categorical ``obs["spatial_frame"]`` names it
  per row (and ``obs["spatial_pixel_frame"]`` the native frame). A table
  mixing calibrated and uncalibrated samples gets no ``spatial`` at all.
- ``uns["spatialdata_io_positions"]`` declares both arrays. An imported
  ``obsm["spatial"]`` of unknown meaning moves unchanged to ``spatial_source``.
- The redundant ``Image`` column is dropped. The legacy feature column
  ``Final Concentration / Dilution`` becomes ``Final_Concentration``. Other
  ``obs`` and ``var`` column names and top-level ``uns`` keys that SpatialData
  does not accept (``Channel Name``, ``my note``) are sanitized with one
  warning per frame or mapping (``Channel_Name``), before the reader adds its
  own columns and keys, which keep their names. ``uns`` values, their order and
  nested keys stay as stored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
from spatialdata.models import TableModel

from spatialdata_io._positions import POSITIONS_KEY, FixedFrame, PositionsMetadata, RowFrames
from spatialdata_io.readers._utils.centroids import scale_regionprops_centroids
from spatialdata_io.readers._utils.errors import add_reader_context
from spatialdata_io.readers._utils.names import sanitize_column_names, sanitize_uns_keys
from spatialdata_io.readers.steinbock._layout import NATIVE_FRAME_SUFFIX
from spatialdata_io.readers.steinbock._schemas import (
    CELL_ID_DTYPE,
    CELL_TABLE_CONTEXT,
    POSITIONS_CONTEXT,
    REQUIRED_OBSERVATION_COLUMNS,
    centroid_columns,
    instance_identifiers,
    metadata_columns,
    validate_centroids,
    validate_position_destinations,
    validate_table,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.steinbock._rasters import SampleGrid

# Columns the reader reads by name or generates; imported columns that equal
# one ignoring letter case are renamed rather than these.
_READER_OBS_COLUMNS = (*REQUIRED_OBSERVATION_COLUMNS, "cell_id", "region", "spatial_frame", "spatial_pixel_frame")
_READER_UNS_KEYS = (POSITIONS_KEY, TableModel.ATTRS_KEY)
_NATIVE_POSITION = "source_regionprops_centroid_zero_based_pixel_index_centres"
_PHYSICAL_POSITION = "source_regionprops_centroid_in_raster_corner_frame"


def read_table(path: ArtifactFile, grids: Mapping[str, SampleGrid]) -> ad.AnnData:
    """Read the table and add identifiers and native centroids without reading mask pixels."""
    try:
        table = ad.read_h5ad(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=CELL_TABLE_CONTEXT, path=path)
        raise

    validate_table(table, path)
    validate_position_destinations(table, multiple=len(grids) > 1, path=path)

    # validate_table established eager frames; AnnData also types backed Dataset2D.
    obs, var = cast("pd.DataFrame", table.obs), cast("pd.DataFrame", table.var)
    columns = metadata_columns(obs, var, path)
    if "Image" in obs:
        del obs["Image"]
    obs.columns = sanitize_column_names(
        list(obs.columns), source="steinbock: the observation metadata of the cell table", reserved=_READER_OBS_COLUMNS
    )
    var.columns = sanitize_column_names(columns, source="steinbock: the feature metadata of the cell table")
    sanitize_uns_keys(table, source="steinbock: the unstructured metadata of the cell table", reserved=_READER_UNS_KEYS)

    if "spatial" in table.obsm:
        table.obsm["spatial_source"] = table.obsm.pop("spatial")
    _set_instance_ids(obs, grids, path)

    coordinates = np.column_stack(centroid_columns(obs, path))
    table.obsm["spatial_pixels"] = coordinates
    _set_positions(table, coordinates, grids, path)
    return table


def _set_instance_ids(obs: pd.DataFrame, grids: Mapping[str, SampleGrid], path: ArtifactFile) -> None:
    """Add the exact ``cell_id`` and the categorical ``region`` of every row."""
    identifiers = instance_identifiers(obs, grids, path)
    obs["cell_id"] = pd.Series(identifiers, index=obs.index, dtype=CELL_ID_DTYPE)
    obs["region"] = pd.Categorical(
        obs["image"].map({grid.image_name: f"{name}_labels" for name, grid in grids.items()})
    )


def _set_positions(table: ad.AnnData, native: np.ndarray, grids: Mapping[str, SampleGrid], path: ArtifactFile) -> None:
    """Declare the native centroids and add physical positions when every sample is calibrated."""
    # read_table produced eager frames and this exact ndarray; AnnData's
    # container annotations additionally admit backed frames and sparse arrays.
    obs = cast("pd.DataFrame", table.obs)
    single = len(grids) == 1
    descriptions: dict[str, FixedFrame | RowFrames] = {}

    if single:
        descriptions["spatial_pixels"] = {
            "axes": ["x", "y"],
            "units": "pixel",
            "frame": f"{next(iter(grids))}{NATIVE_FRAME_SUFFIX}",
            "position": _NATIVE_POSITION,
        }
    else:
        obs["spatial_pixel_frame"] = pd.Categorical(
            obs["image"].map({grid.image_name: f"{name}{NATIVE_FRAME_SUFFIX}" for name, grid in grids.items()})
        )
        descriptions["spatial_pixels"] = {
            "axes": ["x", "y"],
            "units": "pixel",
            "frame_key": "spatial_pixel_frame",
            "position": _NATIVE_POSITION,
        }

    physical = _validate_and_scale_centroids(native, obs, grids, path)
    if physical is not None:
        table.obsm["spatial"] = physical
        evidence = {grid.frame: grid.calibration.source for grid in grids.values() if grid.calibration is not None}
        if single:
            descriptions["spatial"] = {
                "axes": ["x", "y"],
                "units": "micrometer",
                "frame": next(iter(grids.values())).frame,
                "position": _PHYSICAL_POSITION,
                "calibration": evidence,
            }
        else:
            obs["spatial_frame"] = pd.Categorical(
                obs["image"].map({grid.image_name: grid.frame for grid in grids.values()})
            )
            descriptions["spatial"] = {
                "axes": ["x", "y"],
                "units": "micrometer",
                "frame_key": "spatial_frame",
                "position": _PHYSICAL_POSITION,
                "calibration": evidence,
            }

    declaration: PositionsMetadata = {"schema_version": 1, "coordinates": descriptions}
    table.uns[POSITIONS_KEY] = declaration


def _validate_and_scale_centroids(
    native: np.ndarray, obs: pd.DataFrame, grids: Mapping[str, SampleGrid], path: ArtifactFile
) -> np.ndarray | None:
    """Visit each sample's rows once, preserving interleaved observation order."""
    by_image = {grid.image_name: grid for grid in grids.values()}
    calibrated = all(grid.calibration is not None for grid in grids.values())
    physical = np.empty(native.shape, dtype=np.float64) if calibrated else None
    # A slice avoids an index array and a gather for the single-sample layout.
    groups = (
        {next(iter(by_image)): slice(None)}
        if len(by_image) == 1
        else obs.groupby("image", observed=True, sort=False).indices
    )
    for image, rows in groups.items():
        grid = by_image[str(image)]
        values = native[rows]
        validate_centroids(values, grid, path)
        if physical is not None and grid.calibration is not None:
            if isinstance(rows, slice):
                scaled = physical[rows]
                scaled[:] = values
            else:
                # The integer-array gather above owns its values.
                scaled = values.astype(np.float64, copy=False)
            scale_regionprops_centroids(
                scaled, grid.calibration.spacing_xy_um, frame=grid.frame, context=POSITIONS_CONTEXT, path=path
            )
            if not isinstance(rows, slice):
                physical[rows] = scaled
    return physical
