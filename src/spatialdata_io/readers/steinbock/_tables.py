"""Validate Steinbock cell identities, centroid coordinates, and owned metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd

from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers.steinbock._schemas import (
    SIGNED_CELL_ID_DTYPE,
    UNSIGNED_CELL_ID_DTYPE,
    _centroid_columns,
    _instance_identifiers,
    _metadata_columns,
    _validate_centroids,
    _validate_output_column,
    _validate_table,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.steinbock._images import _SampleRasters


def _read_table(path: ArtifactFile, samples: Mapping[str, _SampleRasters]) -> ad.AnnData:
    """Read and enrich an owned table without loading any segmentation pixels."""
    try:
        table = ad.read_h5ad(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=ReaderErrorContext("steinbock", "cell table"), path=path)
        raise

    _validate_table(table, path)

    # The schema establishes eager frames; AnnData also types backed Dataset2D.
    obs, var = cast("pd.DataFrame", table.obs), cast("pd.DataFrame", table.var)

    _set_instance_ids(obs, samples, path)
    table.obsm["spatial"] = _centroids(obs, samples, path)
    _clean_metadata(obs, var, path)

    return table


def _set_instance_ids(obs: pd.DataFrame, samples: Mapping[str, _SampleRasters], path: ArtifactFile) -> None:
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


def _centroids(obs: pd.DataFrame, samples: Mapping[str, _SampleRasters], path: ArtifactFile) -> np.ndarray:
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
