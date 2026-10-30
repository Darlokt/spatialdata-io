"""Validate Steinbock cell identities, centroid coordinates, and owned metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd

from spatialdata_io._positions import POSITIONS_KEY, FixedFrame, PositionsMetadata, RowFrames
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers.steinbock._schemas import (
    SIGNED_CELL_ID_DTYPE,
    UNSIGNED_CELL_ID_DTYPE,
    _centroid_columns,
    _instance_identifiers,
    _metadata_columns,
    _validate_centroids,
    _validate_output_column,
    _validate_position_destinations,
    _validate_table,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.steinbock._images import _SampleRasters
    from spatialdata_io.readers.steinbock._schemas import LabelGrid


def _read_table(path: ArtifactFile, samples: Mapping[str, LabelGrid]) -> ad.AnnData:
    """Read and enrich an owned table without loading any segmentation pixels."""
    try:
        table = ad.read_h5ad(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=ReaderErrorContext("steinbock", "cell table"), path=path)
        raise

    _validate_table(table, path)

    # The schema establishes eager frames; AnnData also types backed Dataset2D.
    obs, var = cast("pd.DataFrame", table.obs), cast("pd.DataFrame", table.var)

    _validate_position_destinations(table, multiple=len(samples) > 1, path=path)
    if "spatial" in table.obsm:
        table.obsm["spatial_source"] = table.obsm.pop("spatial")
    _set_instance_ids(obs, samples, path)
    table.obsm["spatial_pixels"] = _centroids(obs, samples, path)
    _clean_metadata(obs, var, path)

    return table


def _set_instance_ids(obs: pd.DataFrame, samples: Mapping[str, LabelGrid], path: ArtifactFile) -> None:
    """Construct final exact IDs and sample-local region categories in the owned frame."""
    identifiers = _instance_identifiers(obs, samples, path)
    dtype = UNSIGNED_CELL_ID_DTYPE if max(identifiers) > np.iinfo(np.int64).max else SIGNED_CELL_ID_DTYPE
    cell_ids = pd.Series(identifiers, index=obs.index, dtype=dtype)
    regions = pd.Series(
        pd.Categorical(obs["image"].map({f"{name}.tiff": f"{name}_labels" for name in samples})), index=obs.index
    )

    _validate_output_column(obs, "cell_id", cell_ids, path=path)
    _validate_output_column(obs, "region", regions, path=path)

    obs["cell_id"] = cell_ids
    obs["region"] = regions


def _centroids(obs: pd.DataFrame, samples: Mapping[str, LabelGrid], path: ArtifactFile) -> np.ndarray:
    """Extract XY coordinates from regionprops row/column fields."""
    coordinates = np.column_stack(_centroid_columns(obs, path))
    _validate_centroids(coordinates, obs, samples, path)

    return coordinates


def _clean_metadata(obs: pd.DataFrame, var: pd.DataFrame, path: ArtifactFile) -> None:
    """Apply validated export-name cleanup without recasting imported annotations."""
    columns = _metadata_columns(obs, var, path)

    if "Image" in obs:
        del obs["Image"]
    var.columns = columns


def _set_positions(table: ad.AnnData, samples: Mapping[str, _SampleRasters], path: ArtifactFile) -> None:
    """Declare native centroids and construct physical positions from resolved raster state."""
    # _read_table produced eager frames and this exact ndarray; AnnData's
    # container annotations additionally admit backed frames and sparse arrays.
    obs = cast("pd.DataFrame", table.obs)
    native = cast("np.ndarray", table.obsm["spatial_pixels"])
    descriptions: dict[str, FixedFrame | RowFrames] = {}
    native_description: FixedFrame | RowFrames
    if len(samples) == 1:
        native_description = {
            "axes": ["x", "y"],
            "units": "pixel",
            "frame": f"{next(iter(samples))}_regionprops_indices",
            "position": "source_regionprops_centroid_zero_based_pixel_index_centres",
        }
    else:
        obs["spatial_pixel_frame"] = pd.Categorical(
            obs["image"].map({f"{name}.tiff": f"{name}_regionprops_indices" for name in samples})
        )
        native_description = {
            "axes": ["x", "y"],
            "units": "pixel",
            "frame_key": "spatial_pixel_frame",
            "position": "source_regionprops_centroid_zero_based_pixel_index_centres",
        }
    descriptions["spatial_pixels"] = native_description
    spacings = {name: sample.spacing_xy_um for name, sample in samples.items() if sample.spacing_xy_um is not None}
    if len(spacings) == len(samples):
        # Regionprops averages zero-based pixel indices; the unchanged raster
        # locates their centres half a pixel inward from its corner origin.
        with np.errstate(over="ignore", invalid="ignore"):
            physical = np.add(native, 0.5, dtype=np.float64)
            if len(spacings) == 1:
                physical *= next(iter(spacings.values()))
            else:
                for image, rows in obs.groupby("image", observed=True, sort=False).indices.items():
                    physical[rows] *= spacings[str(image).removesuffix(".tiff")]
        if not np.isfinite(physical).all():
            raise ReaderFormatError(
                context=ReaderErrorContext("steinbock", "positions"),
                path=path,
                found="nonfinite physical positions",
                expected="finite float64 physical XY",
            )
        table.obsm["spatial"] = physical
        evidence = {sample.frame: sample.calibration for sample in samples.values()}
        physical_description: FixedFrame | RowFrames
        if len(samples) == 1:
            physical_description = {
                "axes": ["x", "y"],
                "units": "micrometer",
                "frame": next(iter(samples.values())).frame,
                "position": "source_regionprops_centroid_in_raster_corner_frame",
                "calibration": evidence,
            }
        else:
            obs["spatial_frame"] = pd.Categorical(
                obs["image"].map({f"{name}.tiff": sample.frame for name, sample in samples.items()})
            )
            physical_description = {
                "axes": ["x", "y"],
                "units": "micrometer",
                "frame_key": "spatial_frame",
                "position": "source_regionprops_centroid_in_raster_corner_frame",
                "calibration": evidence,
            }
        descriptions["spatial"] = physical_description
    declaration: PositionsMetadata = {"schema_version": 1, "coordinates": descriptions}
    table.uns[POSITIONS_KEY] = declaration
