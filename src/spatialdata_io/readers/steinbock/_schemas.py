"""Steinbock H5AD annotations, compound IDs, centroids, and metadata contracts.

Required observation fields are named and unordered; additional obs/var fields
retain their original storage and missingness. Centroids are finite numeric
pixel coordinates in regionprops YX order. Generated IDs use exact signed or
unsigned 64-bit integers; region IDs are categorical strings. Conflicting
reserved output metadata and ambiguous annotation names are rejected.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Mapping

    import anndata as ad

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.steinbock._images import _SampleRasters

_OBJECT_NAME = re.compile(r"Object ([0-9]+) in (.+)")
REQUIRED_OBSERVATION_COLUMNS = frozenset({"image", "centroid-0", "centroid-1"})
SIGNED_CELL_ID_DTYPE = pd.ArrowDtype(pa.int64())
UNSIGNED_CELL_ID_DTYPE = pd.ArrowDtype(pa.uint64())


def _format_error(path: ArtifactFile, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a cell-table schema failure with the originating artifact."""
    return ReaderFormatError(
        context=ReaderErrorContext("steinbock", "cell table"), path=path, found=found, expected=expected
    )


def _validate_table(table: ad.AnnData, path: ArtifactFile) -> None:
    """Validate imported eager annotations before selecting named source fields."""
    if table.X is None:
        raise _format_error(path, found="missing expression X", expected="an observation-by-feature expression array")
    if not isinstance(table.obs, pd.DataFrame) or not isinstance(table.var, pd.DataFrame):
        message = "Steinbock requires eager DataFrames from read_h5ad."
        raise TypeError(message)

    if not table.obs.columns.is_unique or not table.var.columns.is_unique:
        raise _format_error(
            path, found="duplicate annotation columns", expected="unique observation and feature column names"
        )
    if (
        not table.obs_names.is_unique
        or not table.var_names.is_unique
        or table.obs_names.hasnans
        or table.var_names.hasnans
        or bool(table.var_names.isin([""]).any())
    ):
        raise _format_error(path, found="duplicate or empty identifiers", expected="unique observation and feature IDs")

    obs = table.obs
    required = REQUIRED_OBSERVATION_COLUMNS
    if not required.issubset(obs.columns):
        raise _format_error(path, found=f"columns {obs.columns.tolist()}", expected=f"columns {sorted(required)}")


def _instance_identifiers(obs: pd.DataFrame, samples: Mapping[str, _SampleRasters], path: ArtifactFile) -> list[int]:
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

    return identifiers


def _centroid_columns(obs: pd.DataFrame, path: ArtifactFile) -> tuple[np.ndarray, np.ndarray]:
    """Extract nonnull real XY source arrays without changing imported annotations."""
    columns: list[np.ndarray] = []
    for name in ("centroid-1", "centroid-0"):
        column = obs[name]
        if (
            column.isna().any()
            or pd.api.types.is_bool_dtype(column.dtype)
            or not (pd.api.types.is_integer_dtype(column.dtype) or pd.api.types.is_float_dtype(column.dtype))
        ):
            raise _format_error(path, found="invalid centroids", expected="nonnull real numeric centroid columns")

        # Nullable pandas/Arrow numerics otherwise produce object arrays. Their
        # native dtype preserves width and signedness without a float detour.
        dtype = (
            np.dtype(column.dtype.pyarrow_dtype.to_pandas_dtype())
            if isinstance(column.dtype, pd.ArrowDtype)
            else np.dtype(str(column.dtype).lower())
        )
        columns.append(column.to_numpy(dtype=dtype, copy=False))

    # Combining integer columns with other numeric dtypes can choose floating
    # storage. Check only integers beyond its exact range before conversion.
    common = np.result_type(columns[0].dtype, columns[1].dtype)
    if common.kind == "f":
        precision = np.finfo(common).nmant + 1
        for values in columns:
            if values.dtype.kind in "iu" and np.iinfo(values.dtype).bits > precision:
                limit = 2**precision
                candidates = values > limit
                if values.dtype.kind == "i":
                    candidates |= values < -limit
                if any(int(common.type(value)) != int(value) for value in values[candidates]):
                    raise _format_error(
                        path, found="centroid integer precision loss", expected="lossless common numeric XY storage"
                    )

    return columns[0], columns[1]


def _validate_centroids(
    coordinates: np.ndarray, obs: pd.DataFrame, samples: Mapping[str, _SampleRasters], path: ArtifactFile
) -> None:
    """Require finite XY pixel centers within each cell's independently inspected frame."""
    if not np.isfinite(coordinates).all():
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


def _metadata_columns(obs: pd.DataFrame, var: pd.DataFrame, path: ArtifactFile) -> list[str]:
    """Validate optional redundant image metadata and return collision-free feature keys."""
    if "Image" in obs and not obs["Image"].eq(obs["image"]).fillna(value=False).all():
        raise _format_error(path, found="conflicting Image and image columns", expected="equal source-image metadata")

    if any(not isinstance(name, str) or not name for name in var.columns):
        raise _format_error(path, found="empty or nonstring feature metadata keys", expected="nonempty string keys")
    columns = [
        ("Final Concentration" if name == "Final Concentration / Dilution" else name).replace(" ", "_")
        for name in var.columns
    ]
    if len(set(columns)) != len(columns):
        raise _format_error(path, found="feature metadata cleanup collision", expected="unique normalized column names")

    return columns


def _validate_output_column(obs: pd.DataFrame, key: str, values: pd.Series, *, path: ArtifactFile) -> None:
    """Reject conflicting reserved metadata using exact scalar comparisons.

    Equal existing values are accepted; the generated field takes its declared
    output storage. Python scalar comparison avoids float promotion of uint64 IDs.
    """
    if key in obs and any(
        not pd.api.types.is_scalar(left)
        or pd.isna(left)
        or isinstance(left, bool) != isinstance(right, bool)
        or left != right
        for left, right in zip(obs[key].tolist(), values.tolist(), strict=True)
    ):
        raise _format_error(
            path,
            found=f"conflicting reserved observation column {key!r}",
            expected="absent or semantically equal reader output metadata",
        )
