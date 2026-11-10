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
only Moran statistics are typed, as finite float64. NUL characters are
rejected in every text field because they would let two distinct identifiers
compare equal after truncation.
"""

from __future__ import annotations

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
CLUSTER_TYPES = pa.schema([("bead_id", pa.string()), ("cluster", pa.string())])
DEFINING_FEATURES_KEY = "Top_cluster_defining_features"

RESERVED_OBS = ("cluster", "region", "instance_id")
# uns keys the reader writes; metric categories become uns keys too.
RESERVED_UNS = frozenset({DEFINING_FEATURES_KEY, POSITIONS_KEY, TableModel.ATTRS_KEY})
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

    conflicts = sorted(set(RESERVED_OBS).intersection(table.obs.columns))
    if conflicts or DEFINING_FEATURES_KEY in table.uns:
        raise schema_error(
            COUNTS,
            path,
            found=f"reserved output annotations {conflicts!r} or {DEFINING_FEATURES_KEY!r}",
            expected="H5AD annotations distinct from reader-created output fields",
        )


def validate_header(columns: list[str], artifact: str, path: ArtifactFile, *, feature_ids: bool = False) -> None:
    """Reject empty or NUL-containing names after ``csv_header`` establishes uniqueness.

    With ``feature_ids``, the first column is positional and its name may be empty.
    """
    named = columns[1:] if feature_ids else columns
    if not columns or any(not column for column in named) or any("\x00" in column for column in columns):
        raise schema_error(
            artifact, path, found=f"invalid header {columns!r}", expected="nonempty unique named columns"
        )


def text_types(columns: list[str]) -> pa.Schema:
    """Declare literal strings for the validated named enrichment header."""
    return pa.schema([(name, pa.string()) for name in columns])


def moran_types(columns: list[str]) -> pa.Schema:
    """Declare literal feature IDs followed by float64 statistics for the validated Moran header."""
    return pa.schema([(columns[0], pa.string()), *((name, pa.float64()) for name in columns[1:])])


def validate_text(frame: pd.DataFrame, artifact: str, path: ArtifactFile) -> None:
    """Reject NUL characters in string cells."""
    for column in frame:
        if frame[column].dtype == STRING_DTYPE and frame[column].str.contains("\x00", regex=False).any():
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

    conflicts = sorted(set(metrics["Category"]).intersection(uns_keys | RESERVED_UNS))
    if conflicts:
        raise schema_error(
            METRICS,
            path,
            found=f"conflicting metric categories {conflicts!r}",
            expected="categories distinct from H5AD and reader-created uns keys",
        )


def validate_moran(moran: pd.DataFrame, features: pd.DataFrame, path: ArtifactFile) -> None:
    """Require finite statistics for unique known feature IDs in columns new to ``var``."""
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

    # Arrow parses nan and inf as float64 values; the statistics must be finite.
    if not np.isfinite(moran.to_numpy(dtype=np.float64)).all():
        raise schema_error(
            MORAN_FEATURES, path, found="nonfinite Moran statistics", expected="finite float64 Moran statistics"
        )
