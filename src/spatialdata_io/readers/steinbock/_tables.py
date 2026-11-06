"""Read the cell table (``table_file``) into the linked table and add its identifiers and positions.

The table is read eagerly with AnnData, validated against the schema in
``_schemas`` and then changed in place: the reader owns the freshly read
object, so expression, layers, ``raw`` and annotations are not copied.

- ``obs["cell_id"]`` is the mask label parsed from the observation name
  (int64, or uint64 when a label needs it) and ``obs["region"]`` the
  categorical ``<sample>_labels`` of ``obs["image"]``.
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
- The redundant ``Image`` column is dropped and feature column names are
  cleaned (spaces become underscores; the legacy ``Final Concentration /
  Dilution`` becomes ``Final_Concentration``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd

from spatialdata_io._positions import POSITIONS_KEY, FixedFrame, PositionsMetadata, RowFrames
from spatialdata_io.readers._utils.errors import ReaderFormatError, add_reader_context
from spatialdata_io.readers.steinbock._layout import NATIVE_FRAME_SUFFIX
from spatialdata_io.readers.steinbock._schemas import (
    CELL_TABLE_CONTEXT,
    POSITIONS_CONTEXT,
    SIGNED_CELL_ID_DTYPE,
    UNSIGNED_CELL_ID_DTYPE,
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
    from spatialdata_io.readers.steinbock._images import Calibration, SampleGrid

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
    if "spatial" in table.obsm:
        table.obsm["spatial_source"] = table.obsm.pop("spatial")
    _set_instance_ids(obs, grids, path)

    coordinates = np.column_stack(centroid_columns(obs, path))
    validate_centroids(coordinates, obs, grids, path)
    table.obsm["spatial_pixels"] = coordinates

    columns = metadata_columns(obs, var, path)
    if "Image" in obs:
        del obs["Image"]
    var.columns = columns
    return table


def _set_instance_ids(obs: pd.DataFrame, grids: Mapping[str, SampleGrid], path: ArtifactFile) -> None:
    """Add the exact ``cell_id`` and the categorical ``region`` of every row."""
    identifiers = instance_identifiers(obs, grids, path)
    dtype = UNSIGNED_CELL_ID_DTYPE if max(identifiers) > np.iinfo(np.int64).max else SIGNED_CELL_ID_DTYPE
    obs["cell_id"] = pd.Series(identifiers, index=obs.index, dtype=dtype)
    obs["region"] = pd.Categorical(obs["image"].map({f"{name}.tiff": f"{name}_labels" for name in grids}))


def set_positions(table: ad.AnnData, grids: Mapping[str, SampleGrid], path: ArtifactFile) -> None:
    """Declare the native centroids and add physical positions when every sample is calibrated."""
    # read_table produced eager frames and this exact ndarray; AnnData's
    # container annotations additionally admit backed frames and sparse arrays.
    obs = cast("pd.DataFrame", table.obs)
    native = cast("np.ndarray", table.obsm["spatial_pixels"])
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
            obs["image"].map({f"{name}.tiff": f"{name}{NATIVE_FRAME_SUFFIX}" for name in grids})
        )
        descriptions["spatial_pixels"] = {
            "axes": ["x", "y"],
            "units": "pixel",
            "frame_key": "spatial_pixel_frame",
            "position": _NATIVE_POSITION,
        }

    calibrations = {name: grid.calibration for name, grid in grids.items() if grid.calibration is not None}
    if len(calibrations) == len(grids):
        table.obsm["spatial"] = _physical_positions(native, obs, calibrations, path)
        evidence = {grids[name].frame: calibration.source for name, calibration in calibrations.items()}
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
                obs["image"].map({f"{name}.tiff": grid.frame for name, grid in grids.items()})
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


def _physical_positions(
    native: np.ndarray, obs: pd.DataFrame, calibrations: Mapping[str, Calibration], path: ArtifactFile
) -> np.ndarray:
    """Return float64 micrometre XY, rejecting overflow and nonzero positions that underflow to zero."""
    # Regionprops averages zero-based pixel indices; the unchanged raster
    # locates their centres half a pixel inward from its corner origin.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        physical = np.add(native, 0.5, dtype=np.float64)
        nonzero = physical != 0
        if len(calibrations) == 1:
            physical *= next(iter(calibrations.values())).spacing_xy_um
        else:
            for image, rows in obs.groupby("image", observed=True, sort=False).indices.items():
                physical[rows] *= calibrations[str(image).removesuffix(".tiff")].spacing_xy_um

    if not np.isfinite(physical).all():
        raise ReaderFormatError(
            context=POSITIONS_CONTEXT,
            path=path,
            found="nonfinite physical positions",
            expected="finite float64 physical XY",
        )
    if np.any(nonzero & (physical == 0)):
        raise ReaderFormatError(
            context=POSITIONS_CONTEXT,
            path=path,
            found="nonzero shifted pixel coordinates becoming zero physical positions",
            expected="nonzero float64 physical XY for nonzero shifted pixel coordinates",
        )
    return physical
