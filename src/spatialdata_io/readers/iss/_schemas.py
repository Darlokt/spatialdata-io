"""ISS H5AD input, exact label-ID, and preserved-annotation contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    import anndata as ad

    from spatialdata_io.readers._utils.paths import ArtifactFile

REGION_KEY = "region"
DEFAULT_INSTANCE_KEY = "instance_id"
MIN_INSTANCE_BITS = 16


def _validate_table(table: ad.AnnData, *, path: ArtifactFile, instance_key: str | None = None) -> None:
    """Require expression and unambiguous names while retaining all annotation extensions.

    Obs/var fields may have arbitrary order and additional fields. Observation
    names are nonempty and unique; feature names are nonempty and may repeat
    before the reader's documented uniquification. Imported metadata dtypes and
    missingness remain unchanged. Final linkage rejects empty linked tables.
    """
    if table.X is None:
        raise _schema_error(path, "missing expression X", "an observation-by-feature expression array")
    if not isinstance(table.obs, pd.DataFrame) or not isinstance(table.var, pd.DataFrame):
        message = "ISS requires eager annotation DataFrames from read_h5ad."
        raise TypeError(message)

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


def _validate_index_ids(values: list[int], *, label_dtype: np.dtype[np.generic], path: ArtifactFile) -> None:
    """Validate parsed decimal observation names before integer array construction."""
    maximum = np.iinfo(label_dtype.str).max
    if any(value <= 0 or value > maximum for value in values):
        raise _schema_error(
            path,
            "nonpositive or out-of-range index identifiers",
            f"positive label IDs representable by {label_dtype}. Zero is background",
        )


def _validate_instance_ids(column: pd.Series, *, label_dtype: np.dtype[np.generic], path: ArtifactFile) -> None:
    """Accept nonnull positive integer IDs exactly representable by the label dtype."""
    if (
        column.isna().any()
        or pd.api.types.is_bool_dtype(column.dtype)
        or not pd.api.types.is_integer_dtype(column.dtype)
    ):
        raise _schema_error(
            path, f"null or noninteger identifiers with dtype {column.dtype}", "nonnull positive integer label IDs"
        )
    if len(column) and (int(column.min()) <= 0 or int(column.max()) > np.iinfo(label_dtype.str).max):
        raise _schema_error(
            path,
            "nonpositive or out-of-range instance identifiers",
            f"positive label IDs representable by {label_dtype}. Zero is background",
        )


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
        raise _schema_error(
            path,
            f"conflicting reserved observation column {key!r}",
            "absent or semantically equal reader output metadata",
        )


def _schema_error(path: ArtifactFile, found: str, expected: str) -> ReaderFormatError:
    """Describe an external table contract failure."""
    return ReaderFormatError(context=ReaderErrorContext("iss", "counts"), path=path, found=found, expected=expected)
