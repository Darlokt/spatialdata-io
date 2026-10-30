"""Curio enrichment contracts and lossless Arrow normalization.

Clusters are exactly two positional fields. Metrics retain positional metric
and value fields with arbitrary names, followed by named Category and optional
metadata. Defining-feature columns are extensible named annotations; Moran
columns after the first feature-ID field are dynamic finite numeric or boolean
statistics. Imported H5AD annotations retain their original representations.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    import anndata as ad

    from spatialdata_io.readers._utils.paths import ArtifactFile

STRING_DTYPE = pd.ArrowDtype(pa.string())
_INTEGER_DTYPES = {"i": pd.ArrowDtype(pa.int64()), "u": pd.ArrowDtype(pa.uint64())}
FLOAT_DTYPE = pd.ArrowDtype(pa.float64())
BOOLEAN_DTYPE = pd.ArrowDtype(pa.bool_())
OBS_COLUMNS = ("cluster", "region", "instance_id")
DEFINING_FEATURES_KEY = "Top_cluster_defining_features"
_CLUSTER_FIELDS = 2
_MIN_METRIC_FIELDS = 3


def schema_error(path: ArtifactFile, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a malformed table artifact with its validated source path."""
    return ReaderFormatError(context=ReaderErrorContext("curio", path.name), path=path, found=found, expected=expected)


def validate_counts(table: ad.AnnData, path: ArtifactFile) -> None:
    """Require owned count-bearing H5AD with unambiguous, nonconflicting annotations."""
    if table.X is None:
        raise schema_error(
            path,
            found="missing expression in AnnData.X",
            expected="an observation-by-feature expression array in AnnData.X",
        )

    if table.n_obs == 0 or any(
        not identifiers.is_unique or identifiers.hasnans or identifiers.isin([""]).any()
        for identifiers in (table.obs_names, table.var_names)
    ):
        raise schema_error(
            path,
            found="empty observations or missing, empty, or duplicate identifiers",
            expected="nonempty unique observation and feature identifiers",
        )

    if any(not frame.columns.is_unique or frame.columns.hasnans for frame in (table.obs, table.var)):
        raise schema_error(
            path, found="duplicate or missing annotation column names", expected="unique non-null H5AD annotation names"
        )

    conflicts = set(OBS_COLUMNS).intersection(table.obs.columns)
    if conflicts or DEFINING_FEATURES_KEY in table.uns:
        raise schema_error(
            path,
            found=f"reserved output annotations {sorted(conflicts)!r} or {DEFINING_FEATURES_KEY!r}",
            expected="H5AD annotations distinct from reader-created output fields",
        )


def validate_header(columns: list[str], path: ArtifactFile, *, feature_ids: bool = False) -> None:
    """Reject ambiguous named columns before a parser can repair them.

    The first Moran column is a positional feature ID and may have an empty
    R-style row-name header. Every other field requires a unique nonempty name.
    """
    named = columns[1:] if feature_ids else columns
    if (
        not columns
        or any(not column for column in named)
        or any("\x00" in column for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise schema_error(path, found="invalid enrichment header", expected="nonempty unique named columns")


def normalize_metadata(frame: pd.DataFrame, path: ArtifactFile, *, strings: tuple[str, ...] = ()) -> pd.DataFrame:
    """Normalize extensible primitive annotation columns without changing source order.

    Declared identifiers stay lexical strings. Other columns accept integer,
    floating, boolean, or string tokens. Integer storage is signed/unsigned
    64-bit; floating text uses Arrow's correctly rounded parser. Large numeric
    values and underflow must remain exactly representable. Nulls are retained.
    """
    for column in frame:
        source = frame[column]
        if source.str.contains("\x00", regex=False).any():
            raise schema_error(
                path, found=f"NUL character in {column!r}", expected="text fields without NUL characters"
            )
        if column in strings:
            continue

        frame[column] = _normalize_primitive(source, path)

    return frame


def _normalize_primitive(source: pd.Series, path: ArtifactFile) -> pd.Series:
    """Choose one explicit primitive dtype while preserving nonnumeric lexical metadata."""
    present = source.dropna()
    if len(present) and present.isin(["TRUE", "FALSE", "True", "False", "true", "false"]).all():
        return source.map(
            {"TRUE": True, "FALSE": False, "True": True, "False": False, "true": True, "false": False}
        ).astype(BOOLEAN_DTYPE)

    try:
        numeric = pd.to_numeric(source)
    except ValueError:
        return source
    if numeric.dtype.kind in _INTEGER_DTYPES:
        return numeric.astype(_INTEGER_DTYPES[numeric.dtype.kind])

    try:
        numeric = source.astype(FLOAT_DTYPE)
    except (pa.ArrowInvalid, ValueError):
        return source

    values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    nonfinite = np.flatnonzero(~np.isfinite(values) & source.notna().to_numpy())
    if any(Decimal(source.iloc[row]).is_finite() for row in nonfinite):
        return source

    candidates = np.isfinite(values) & ((np.abs(values) >= 2**53) | (values == 0)) & source.notna().to_numpy()
    for row in np.flatnonzero(candidates):
        if Decimal(source.iloc[row]) != Decimal(float(values[row])):
            raise schema_error(
                path,
                found=f"numeric precision loss in {source.name!r}",
                expected="losslessly representable numeric values",
            )

    return numeric


def validate_clusters(clusters: pd.DataFrame, observations: pd.Index, path: ArtifactFile) -> None:
    """Require one two-field cluster assignment per exact observation identifier."""
    if clusters.shape[1] != _CLUSTER_FIELDS or clusters.isna().any(axis=None) or clusters.isin([""]).any(axis=None):
        raise schema_error(
            path, found="malformed cluster assignments", expected="two non-null columns: bead ID and cluster"
        )

    identifiers = pd.Index(clusters.iloc[:, 0])
    if not identifiers.is_unique or set(identifiers) != set(observations):
        raise schema_error(
            path,
            found="duplicate or mismatched bead IDs",
            expected="exactly one cluster assignment per H5AD observation",
        )


def validate_metrics(metrics: pd.DataFrame, path: ArtifactFile, uns_keys: set[str]) -> None:
    """Validate the positional metric/value exception and named trailing categories."""
    if len(metrics.columns) < _MIN_METRIC_FIELDS or "Category" not in metrics.columns[2:]:
        raise schema_error(
            path,
            found=f"columns {metrics.columns.tolist()!r}",
            expected="metric and value columns followed by Category, with matching row widths",
        )

    key = metrics.columns[0]
    identifiers = metrics[[key, "Category"]]
    if (
        identifiers.isna().any(axis=None)
        or identifiers.isin([""]).any(axis=None)
        or metrics.duplicated(["Category", key]).any()
    ):
        raise schema_error(
            path,
            found="empty or duplicate metric identifiers",
            expected="nonempty categories and unique metric names within each category",
        )
    if set(metrics["Category"]).intersection(uns_keys | {DEFINING_FEATURES_KEY}):
        raise schema_error(
            path, found="conflicting metric categories", expected="categories distinct from existing uns keys"
        )


def validate_moran(moran: pd.DataFrame, features: pd.DataFrame, path: ArtifactFile) -> None:
    """Require unique known feature IDs and finite dynamic numeric/boolean statistics."""
    if moran.shape[1] == 0:
        raise schema_error(path, found="missing Moran statistics", expected="feature IDs followed by Moran statistics")
    identifiers = moran.index
    if (
        not identifiers.is_unique
        or identifiers.hasnans
        or identifiers.isin([""]).any()
        or not identifiers.isin(features.index).all()
    ):
        raise schema_error(
            path,
            found="duplicate, empty, or unknown feature IDs",
            expected="unique feature IDs contained in H5AD var_names",
        )
    if len(moran.columns.intersection(features.columns)):
        raise schema_error(
            path,
            found="overlapping H5AD variable columns",
            expected="Moran statistic columns distinct from existing variable metadata",
        )
    if len(moran) and any(
        not (pd.api.types.is_numeric_dtype(moran[column].dtype) or pd.api.types.is_bool_dtype(moran[column].dtype))
        or moran[column].isna().any()
        or not np.isfinite(moran[column].to_numpy()).all()
        for column in moran
    ):
        raise schema_error(
            path,
            found="nonnumeric or nonfinite Moran statistics",
            expected="finite numeric or boolean statistics for each listed feature",
        )
