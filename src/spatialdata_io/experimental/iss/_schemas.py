"""Input and output contracts of the caller-selected ISS ``.h5ad`` table.

Input: one AnnData file read eagerly. Observations are cells and features are
genes or targets::

    X                    required expression array (dense or CSR/CSC sparse)
    layers["counts"]     optional; normalized like X. Other layers are kept as is
    obs_names            nonempty, unique; decimal ASCII label IDs unless instance_key is given
    var_names            nonempty; duplicates are made unique (GeneB, GeneB-1, ...)
    obs[instance_key]    optional; non-null positive integers
    obsm["spatial"]      optional; moved unchanged to obsm["spatial_source"]

Obs and var fields may come in any order with additional fields, and imported
dtypes and missingness are kept; duplicate field names are rejected. Label IDs
are positive (zero is label background) and must fit the label image dtype.

Output: ``obs["region"]`` (categorical labels element name) and the instance
column (``instance_key``, or ``instance_id`` built from the observation names)
link each row to the labels. A source field already named like an output column
must hold equal values. ``obsm["spatial_source"]`` and
``uns["spatialdata_io_positions"]`` are reserved and must be absent, because no
axes or units are known for the imported embedding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    import anndata as ad

    from spatialdata_io.readers._utils.paths import ArtifactFile

REGION_KEY = "region"
DEFAULT_INSTANCE_KEY = "instance_id"
# Linkage requires at least 16-bit integer instance IDs.
MIN_INSTANCE_BITS = 16
TABLE_CONTEXT = ReaderErrorContext("iss", "table")


def validate_table(table: ad.AnnData, *, path: ArtifactFile, instance_key: str | None) -> None:
    """Require expression and unambiguous names while keeping all annotation extensions."""
    if POSITIONS_KEY in table.uns:
        raise _schema_error(path, f"occupied {POSITIONS_KEY} declaration", "an unannotated source embedding")
    if "spatial" in table.obsm and "spatial_source" in table.obsm:
        raise _schema_error(path, "occupied obsm[spatial_source]", "an unoccupied imported-coordinate destination")
    if table.X is None:
        raise _schema_error(path, "missing expression X", "an observation-by-feature expression array")

    if not table.obs.columns.is_unique or not table.var.columns.is_unique:
        raise _schema_error(path, "duplicate annotation columns", "unique observation and feature column names")
    if (
        not table.obs_names.is_unique
        or table.obs_names.hasnans
        or table.var_names.hasnans
        or bool(table.obs_names.isin([""]).any())
        or bool(table.var_names.isin([""]).any())
    ):
        raise _schema_error(
            path,
            "duplicate observations or empty observation/feature identifiers",
            "unique nonempty observations and nonempty features",
        )

    if instance_key is not None and instance_key not in table.obs:
        raise _schema_error(
            path, f"missing observation column {instance_key!r}", "the requested integer instance column"
        )


def validate_index_ids(values: list[int], *, label_dtype: np.dtype[np.integer], path: ArtifactFile) -> None:
    """Validate IDs parsed from observation names before building an integer array."""
    maximum = np.iinfo(label_dtype).max
    if any(value <= 0 or value > maximum for value in values):
        raise _schema_error(
            path,
            "nonpositive or out-of-range index identifiers",
            f"positive label IDs representable by {label_dtype}. Zero is background",
        )


def validate_instance_ids(column: pd.Series, *, label_dtype: np.dtype[np.integer], path: ArtifactFile) -> None:
    """Accept non-null positive integer IDs representable by the label dtype."""
    if (
        column.isna().any()
        or pd.api.types.is_bool_dtype(column.dtype)
        or not pd.api.types.is_integer_dtype(column.dtype)
    ):
        raise _schema_error(
            path, f"null or noninteger identifiers with dtype {column.dtype}", "nonnull positive integer label IDs"
        )
    if len(column) and (int(column.min()) <= 0 or int(column.max()) > np.iinfo(label_dtype).max):
        raise _schema_error(
            path,
            "nonpositive or out-of-range instance identifiers",
            f"positive label IDs representable by {label_dtype}. Zero is background",
        )


def validate_output_column(obs: pd.DataFrame, key: str, values: pd.Series, *, path: ArtifactFile) -> None:
    """Reject a source field that conflicts with a generated output column.

    Equal existing values are accepted and replaced by the generated storage.
    Python scalar comparison avoids float promotion of uint64 IDs.
    """
    if key in obs and any(
        not pd.api.types.is_scalar(left)
        or pd.isna(left)
        or isinstance(left, bool) != isinstance(right, bool)
        or left != right
        for left, right in zip(obs[key].tolist(), values.tolist(), strict=True)
    ):
        raise _schema_error(
            path,
            f"conflicting reserved observation column {key!r}",
            "absent or semantically equal reader output metadata",
        )


def _schema_error(path: ArtifactFile, found: str, expected: str) -> ReaderFormatError:
    """Describe an external table contract failure."""
    return ReaderFormatError(context=TABLE_CONTEXT, path=path, found=found, expected=expected)
