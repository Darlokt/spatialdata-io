"""Column contracts of the Curio H5AD and its four text enrichments.

The H5AD supplies bead observations, feature IDs and expression in their final
order. The four text files are aligned to it by exact identifier:

- ``*_cluster_assignment.txt``: headerless TSV with exactly two positional
  fields, bead ID and cluster label, one row per H5AD observation in any order.
- ``*_Metrics.csv``: CSV whose first two columns are the metric name and its
  value under arbitrary header names, followed by a named ``Category`` column
  and optional further columns, which are ignored. Metric names are unique
  within a category.
- ``*_variable_features_clusters.txt``: TSV of named columns, kept whole.
- ``*_variable_features_spatial_moransi.txt``: TSV whose first column holds
  feature IDs (its header may be empty, as R writes row names) and whose other
  named columns hold one statistic each. Listed IDs are unique H5AD features.

Every header name must be nonempty and unique so that pandas cannot rename
columns silently. Cells are kept as exact strings, without type inference;
only Moran statistics are parsed, as exact integers, finite float64 or
booleans. NUL characters are rejected everywhere because they would let two
distinct identifiers compare equal after truncation.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
from spatialdata.models import TableModel

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers.curio._layout import CLUSTERS, COUNTS, METRICS, MORAN_FEATURES

if TYPE_CHECKING:
    import anndata as ad

    from spatialdata_io.readers._utils.paths import ArtifactFile

STRING_DTYPE = pd.ArrowDtype(pa.string())
CLUSTER_COLUMNS = ["bead_id", "cluster"]
DEFINING_FEATURES_KEY = "Top_cluster_defining_features"

_INTEGER_DTYPES = {"i": pd.ArrowDtype(pa.int64()), "u": pd.ArrowDtype(pa.uint64())}
_FLOAT_DTYPE = pd.ArrowDtype(pa.float64())
_BOOLEAN_DTYPE = pd.ArrowDtype(pa.bool_())
_BOOLEANS = {"TRUE": True, "FALSE": False, "True": True, "False": False, "true": True, "false": False}
_RESERVED_OBS = ("cluster", "region", "instance_id")
# uns keys the reader writes; metric categories become uns keys too.
_RESERVED_UNS = frozenset({DEFINING_FEATURES_KEY, POSITIONS_KEY, TableModel.ATTRS_KEY})
_MIN_METRIC_FIELDS = 3


def schema_error(artifact: str, path: ArtifactFile, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a malformed Curio artifact."""
    return ReaderFormatError(context=ReaderErrorContext("curio", artifact), path=path, found=found, expected=expected)


def validate_counts(table: ad.AnnData, path: ArtifactFile) -> None:
    """Require expression, unambiguous identifiers and free reader destinations."""
    if table.X is None:
        raise schema_error(
            COUNTS,
            path,
            found="missing expression in AnnData.X",
            expected="an observation-by-feature expression array in AnnData.X",
        )

    if table.n_obs == 0 or any(
        not identifiers.is_unique or identifiers.hasnans or identifiers.isin([""]).any()
        for identifiers in (table.obs_names, table.var_names)
    ):
        raise schema_error(
            COUNTS,
            path,
            found="empty observations or missing, empty, or duplicate identifiers",
            expected="nonempty unique observation and feature identifiers",
        )

    if any(not frame.columns.is_unique or frame.columns.hasnans for frame in (table.obs, table.var)):
        raise schema_error(
            COUNTS,
            path,
            found="duplicate or missing annotation column names",
            expected="unique non-null H5AD annotation names",
        )

    conflicts = sorted(set(_RESERVED_OBS).intersection(table.obs.columns))
    if conflicts or DEFINING_FEATURES_KEY in table.uns:
        raise schema_error(
            COUNTS,
            path,
            found=f"reserved output annotations {conflicts!r} or {DEFINING_FEATURES_KEY!r}",
            expected="H5AD annotations distinct from reader-created output fields",
        )


def validate_header(columns: list[str], artifact: str, path: ArtifactFile, *, feature_ids: bool = False) -> None:
    """Reject empty, duplicate or NUL-containing header names before pandas can repair them.

    With ``feature_ids``, the first column is positional and its name may be empty.
    """
    named = columns[1:] if feature_ids else columns
    if (
        not columns
        or any(not column for column in named)
        or any("\x00" in column for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise schema_error(
            artifact, path, found=f"invalid header {columns!r}", expected="nonempty unique named columns"
        )


def validate_text(frame: pd.DataFrame, artifact: str, path: ArtifactFile) -> None:
    """Reject NUL characters in string cells."""
    for column in frame:
        if frame[column].str.contains("\x00", regex=False).any():
            raise schema_error(
                artifact, path, found=f"NUL character in {column!r}", expected="text fields without NUL characters"
            )


def validate_clusters(clusters: pd.DataFrame, observations: pd.Index, path: ArtifactFile) -> None:
    """Require exactly one nonempty cluster label per H5AD observation."""
    if clusters.isin([""]).any(axis=None):
        raise schema_error(
            CLUSTERS, path, found="empty bead ID or cluster label", expected="nonempty bead IDs and cluster labels"
        )

    identifiers = pd.Index(clusters["bead_id"])
    if not identifiers.is_unique or set(identifiers) != set(observations):
        raise schema_error(
            CLUSTERS,
            path,
            found="duplicate or mismatched bead IDs",
            expected="exactly one cluster assignment per H5AD observation",
        )


def validate_metrics(metrics: pd.DataFrame, path: ArtifactFile, uns_keys: set[str]) -> None:
    """Require positional metric and value columns, a named Category and unique metric keys."""
    if len(metrics.columns) < _MIN_METRIC_FIELDS or "Category" not in metrics.columns[2:]:
        raise schema_error(
            METRICS,
            path,
            found=f"columns {metrics.columns.tolist()!r}",
            expected="metric and value columns followed by Category",
        )

    key = metrics.columns[0]
    identifiers = metrics[[key, "Category"]]
    if identifiers.isin([""]).any(axis=None) or metrics.duplicated(["Category", key]).any():
        raise schema_error(
            METRICS,
            path,
            found="empty or duplicate metric identifiers",
            expected="nonempty categories and unique metric names within each category",
        )

    conflicts = sorted(set(metrics["Category"]).intersection(uns_keys | _RESERVED_UNS))
    if conflicts:
        raise schema_error(
            METRICS,
            path,
            found=f"conflicting metric categories {conflicts!r}",
            expected="categories distinct from H5AD and reader-created uns keys",
        )


def validate_moran(moran: pd.DataFrame, features: pd.DataFrame, path: ArtifactFile) -> None:
    """Require statistics for unique known feature IDs in columns new to ``var``."""
    if moran.shape[1] == 0:
        raise schema_error(
            MORAN_FEATURES, path, found="missing Moran statistics", expected="feature IDs followed by Moran statistics"
        )

    identifiers = moran.index
    if not identifiers.is_unique or identifiers.isin([""]).any() or not identifiers.isin(features.index).all():
        raise schema_error(
            MORAN_FEATURES,
            path,
            found="duplicate, empty, or unknown feature IDs",
            expected="unique feature IDs contained in H5AD var_names",
        )

    if len(moran.columns.intersection(features.columns)):
        raise schema_error(
            MORAN_FEATURES,
            path,
            found="overlapping H5AD variable columns",
            expected="Moran statistic columns distinct from existing variable metadata",
        )


def normalize_moran(moran: pd.DataFrame, path: ArtifactFile) -> None:
    """Parse every statistic column in place."""
    for column in moran:
        moran[column] = _parse_statistic(moran[column], path)


def _parse_statistic(source: pd.Series, path: ArtifactFile) -> pd.Series:
    """Parse one column as booleans, exact 64-bit integers or finite float64."""
    if len(source) and source.isin(list(_BOOLEANS)).all():
        return source.map(_BOOLEANS).astype(_BOOLEAN_DTYPE)

    try:
        numeric = pd.to_numeric(source)
    except ValueError as error:
        raise schema_error(
            MORAN_FEATURES,
            path,
            found=f"nonnumeric Moran statistic {source.name!r}",
            expected="finite numeric or boolean statistics",
        ) from error
    if numeric.dtype.kind in _INTEGER_DTYPES:
        return numeric.astype(_INTEGER_DTYPES[numeric.dtype.kind])

    # Parse the source text again with Arrow, whose cast is correctly rounded;
    # pandas turns "0.10000000000000002" into 0.1 and returns objects beyond 64 bits.
    try:
        numeric = source.astype(_FLOAT_DTYPE)
    except (pa.ArrowInvalid, ValueError) as error:
        raise schema_error(
            MORAN_FEATURES,
            path,
            found=f"invalid Moran statistic {source.name!r}",
            expected="finite numeric or boolean statistics",
        ) from error

    values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(values).all():
        raise schema_error(
            MORAN_FEATURES,
            path,
            found=f"nonfinite Moran statistic {source.name!r}",
            expected="finite numeric or boolean statistics",
        )

    # Decimal fractions round by nature; integral spellings beyond 2**53 and
    # underflow to zero would silently change the value, so only those are checked.
    candidates = (np.abs(values) >= 2**53) | (values == 0)
    for row in np.flatnonzero(candidates):
        if Decimal(source.iloc[row]) != Decimal(float(values[row])):
            raise schema_error(
                MORAN_FEATURES,
                path,
                found=f"numeric precision loss in {source.name!r}",
                expected="losslessly representable numeric values",
            )

    return numeric
