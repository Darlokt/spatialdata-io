"""Validate Steinbock cell identities, centroid coordinates, and owned metadata."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.steinbock._images import _SampleRasters

_OBJECT_NAME = re.compile(r"Object ([0-9]+) in (.+)")


def _format_error(path: ArtifactFile, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a cell-table schema failure with the originating artifact."""
    return ReaderFormatError(
        context=ReaderErrorContext("steinbock", "cell table"), path=path, found=found, expected=expected
    )


def _read_table(path: ArtifactFile, samples: Mapping[str, _SampleRasters]) -> ad.AnnData:
    """Read and enrich an owned table without loading any segmentation pixels."""
    try:
        table = ad.read_h5ad(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=ReaderErrorContext("steinbock", "cell table"), path=path)
        raise
    if table.X is None:
        raise _format_error(path, found="missing expression X", expected="an observation-by-feature expression array")
    if not isinstance(table.obs, pd.DataFrame) or not isinstance(table.var, pd.DataFrame):
        message = "Steinbock requires eager DataFrames from read_h5ad."
        raise TypeError(message)
    if not table.obs_names.is_unique or not table.var_names.is_unique or bool(table.var_names.isin([""]).any()):
        raise _format_error(path, found="duplicate or empty identifiers", expected="unique observation and feature IDs")
    obs = table.obs
    required = {"image", "centroid-0", "centroid-1"}
    if not required.issubset(obs.columns):
        raise _format_error(path, found=f"columns {obs.columns.tolist()}", expected=f"columns {sorted(required)}")
    _set_instance_ids(obs, samples, path)
    table.obsm["spatial"] = _centroids(obs, samples, path)
    _clean_metadata(table.obs, table.var, path)
    return table


def _set_instance_ids(obs: pd.DataFrame, samples: Mapping[str, _SampleRasters], path: ArtifactFile) -> None:
    """Parse exact compound identifiers and require every discovered sample in the table."""
    source_names = {f"{name}.tiff": name for name in samples}
    if obs["image"].isna().any() or set(obs["image"]) != set(source_names):
        raise _format_error(
            path, found="table/image/mask sample disagreement", expected="exactly the discovered sample set"
        )

    # The upstream compound name includes the complete filename, including spaces.
    # Parse decimal IDs directly: a float intermediate cannot retain uint64 values.
    maxima = {f"{name}.tiff": np.iinfo(sample.label_dtype.str).max for name, sample in samples.items()}
    identifiers: list[int] = []
    for observation, source_name in zip(obs.index, obs["image"], strict=True):
        match = _OBJECT_NAME.fullmatch(observation)
        if match is None or match[2] != source_name:
            raise _format_error(
                path, found=f"observation {observation!r}", expected="Object <positive integer> in <obs.image>"
            )
        identifier = int(match[1])
        if identifier <= 0 or identifier > maxima[source_name]:
            raise _format_error(
                path,
                found=f"label ID {identifier} for {source_name!r}",
                expected="a positive ID fitting its mask dtype",
            )
        identifiers.append(identifier)
    dtype = np.uint64 if max(identifiers) > np.iinfo(np.int64).max else np.int64
    obs["cell_id"] = np.asarray(identifiers, dtype=dtype)
    obs["region"] = pd.Categorical(
        obs["image"].map({source: f"{name}_labels" for source, name in source_names.items()})
    )


def _centroids(obs: pd.DataFrame, samples: Mapping[str, _SampleRasters], path: ArtifactFile) -> np.ndarray:
    """Return finite XY pixel centers within each cell's independently inspected frame."""
    # regionprops centroids are (row, column); SpatialData coordinates are (x, y).
    coordinates = obs[["centroid-1", "centroid-0"]].to_numpy()
    if coordinates.dtype.kind not in "iuf" or not np.isfinite(coordinates).all():
        raise _format_error(path, found="invalid centroids", expected="finite real centroid coordinates")
    # Group once rather than rescanning the entire cell population for each image.
    for filename, rows in obs.groupby("image", sort=False, observed=True).indices.items():
        positions = coordinates[rows]
        height, width = samples[str(filename).removesuffix(".tiff")].label_shape
        if (positions < 0).any() or (positions[:, 0] > width - 1).any() or (positions[:, 1] > height - 1).any():
            raise _format_error(
                path,
                found=f"centroids outside {filename!r}",
                expected=f"pixel centers within YX extent {(height, width)}",
            )
    return coordinates


def _clean_metadata(obs: pd.DataFrame, var: pd.DataFrame, path: ArtifactFile) -> None:
    """Clean documented export metadata in place without losing conflicting fields."""
    if "Image" in obs:
        if not obs["Image"].eq(obs["image"]).all():
            raise _format_error(
                path, found="conflicting Image and image columns", expected="equal source-image metadata"
            )
        del obs["Image"]
    if any(not isinstance(name, str) or not name for name in var.columns):
        raise _format_error(path, found="empty or nonstring feature metadata keys", expected="nonempty string keys")
    columns = [
        ("Final Concentration" if name == "Final Concentration / Dilution" else name).replace(" ", "_")
        for name in var.columns
    ]
    if len(set(columns)) != len(columns):
        raise _format_error(path, found="feature metadata cleanup collision", expected="unique normalized column names")
    var.columns = columns
