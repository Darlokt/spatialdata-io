"""Load the Curio H5AD and align the four text enrichments to it by identifier.

``read_h5ad`` loads the whole H5AD eagerly; the reader owns the result and
enriches it in place. Observation and feature order, ``X`` and every imported
slot stay as stored. The text files add:

- ``obs["cluster"]``: categorical cluster labels aligned by bead ID, with
  categories in order of first occurrence along the H5AD rows;
- ``uns[<Category>]``: one ``{metric: value}`` dictionary per metric category;
  a category that is not a valid ``uns`` key (``Run Metrics``) or equals an
  H5AD ``uns`` key ignoring letter case is sanitized with one warning;
- ``uns["Top_cluster_defining_features"]``: the cluster-features table, with
  empty cells as missing values;
- one ``var`` column per Moran statistic, joined by feature ID, missing for
  features the file does not list. Statistic names that a SpatialData table
  does not accept (``MoransI observed``) are sanitized with one warning.

H5AD ``obs`` and ``var`` column names and top-level ``uns`` keys that a
SpatialData table does not accept (``my note``) are sanitized with one warning
per frame or mapping before the reader adds its own columns and keys, which
keep their names; ``uns`` values, their order and nested keys stay as stored.
Metric categories are then sanitized against the sanitized ``uns`` keys.

Text files are UTF-8, optionally with a byte-order mark. pandas' Python parser
is used because it keeps NUL characters for rejection, distinguishes a missing
field from an empty one and rejects malformed quoting; blank lines are skipped.
The files are small (one row per bead, metric or listed feature).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anndata as ad
import pandas as pd

from spatialdata_io.readers._utils.csv import read_csv_header
from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers._utils.names import sanitize_column_names, sanitize_uns_keys
from spatialdata_io.readers.curio._layout import CLUSTER_FEATURES, CLUSTERS, COUNTS, METRICS, MORAN_FEATURES
from spatialdata_io.readers.curio._schemas import (
    CLUSTER_COLUMNS,
    DEFINING_FEATURES_KEY,
    RESERVED_OBS,
    RESERVED_UNS,
    STRING_DTYPE,
    normalize_moran,
    schema_error,
    validate_clusters,
    validate_counts,
    validate_header,
    validate_metrics,
    validate_moran,
    validate_text,
)

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.curio._layout import CurioLayout


def read_table(layout: CurioLayout) -> ad.AnnData:
    """Load the H5AD and attach clusters, metrics, defining features and Moran statistics."""
    try:
        table = ad.read_h5ad(layout.counts)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=ReaderErrorContext("curio", COUNTS), path=layout.counts)
        raise
    validate_counts(table, layout.counts)
    # read_h5ad returns eager DataFrames; AnnData's annotation also includes backed Dataset2D.
    obs, var = cast("pd.DataFrame", table.obs), cast("pd.DataFrame", table.var)
    obs.columns = sanitize_column_names(list(obs.columns), source="curio: the H5AD obs", reserved=RESERVED_OBS)
    var.columns = sanitize_column_names(list(var.columns), source="curio: the H5AD var")
    sanitize_uns_keys(table, source="curio: the H5AD uns", reserved=RESERVED_UNS)

    clusters = _read_rows(layout.clusters, CLUSTERS, delimiter="\t", columns=CLUSTER_COLUMNS, header=False)
    validate_clusters(clusters, table.obs_names, layout.clusters)
    labels = clusters.set_index("bead_id")["cluster"].reindex(table.obs_names)
    table.obs["cluster"] = pd.Categorical(labels, categories=labels.unique())

    _add_metrics(table, layout.metrics)

    defining = _read_named_rows(layout.cluster_features, CLUSTER_FEATURES, delimiter="\t")
    table.uns[DEFINING_FEATURES_KEY] = defining.replace("", pd.NA)

    table.var = _join_moran(var, layout.moran_features)
    return table


def _add_metrics(table: ad.AnnData, path: ArtifactFile) -> None:
    """Store each metric category as a ``{metric: value}`` dictionary in source order."""
    metrics = _read_named_rows(path, METRICS, delimiter=",")
    validate_metrics(metrics, path, set(table.uns))

    # Categories become top-level uns keys, which SpatialData checks like element names.
    categories = list(dict.fromkeys(metrics["Category"].tolist()))
    keys = sanitize_column_names(
        categories,
        source="curio: the Category column of the metrics file",
        reserved=[*table.uns.keys(), *RESERVED_UNS],
        kind="uns",
    )
    key_of = dict(zip(categories, keys, strict=True))
    name, value = metrics.columns[:2]
    for category, group in metrics.groupby("Category", sort=False):
        table.uns[key_of[category]] = dict(zip(group[name], group[value], strict=True))


def _join_moran(features: pd.DataFrame, path: ArtifactFile) -> pd.DataFrame:
    """Join parsed Moran statistics onto ``var`` by feature ID, keeping feature order."""
    moran = _read_named_rows(path, MORAN_FEATURES, delimiter="\t", feature_ids=True)
    moran = moran.set_index(moran.columns[0])
    validate_moran(moran, features, path)
    normalize_moran(moran, path)
    moran.columns = sanitize_column_names(
        list(moran.columns), source=f"curio: the {MORAN_FEATURES} header", reserved=list(features.columns)
    )

    # Match the H5AD index representation so the join keeps it.
    moran.index = moran.index.astype(features.index.dtype)
    return features.join(moran, how="left", validate="one_to_one")


def _read_named_rows(path: ArtifactFile, artifact: str, *, delimiter: str, feature_ids: bool = False) -> pd.DataFrame:
    """Read a text table whose header names its columns."""
    columns = read_csv_header(path, context=ReaderErrorContext("curio", artifact), delimiter=delimiter)
    validate_header(columns, artifact, path, feature_ids=feature_ids)
    return _read_rows(path, artifact, delimiter=delimiter, columns=columns, header=True)


def _read_rows(path: ArtifactFile, artifact: str, *, delimiter: str, columns: list[str], header: bool) -> pd.DataFrame:
    """Read every cell as an exact string, requiring one field per column in every row."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            if header:
                # The header was parsed strictly from the first physical line.
                source.readline()

            # pandas fails on input without records when index_col=[], so build the empty frame here.
            position = source.tell()
            if not source.read(1):
                return pd.DataFrame({column: pd.Series([], dtype=STRING_DTYPE) for column in columns})
            source.seek(position)

            frame = pd.read_csv(
                source,
                sep=delimiter,
                names=columns,
                header=None,
                index_col=[],
                dtype=STRING_DTYPE,
                keep_default_na=False,
                engine="python",
            )
    except OSError as error:
        add_reader_context(error, context=ReaderErrorContext("curio", artifact), path=path)
        raise
    except UnicodeDecodeError as error:
        raise schema_error(artifact, path, found="text that is not valid UTF-8", expected="UTF-8 text") from error
    except pd.errors.ParserError as error:
        raise schema_error(
            artifact, path, found=f"malformed rows ({error})", expected="one field per column in every row"
        ) from error

    # Short rows are padded with missing values; longer rows raise ParserError above.
    if frame.isna().any(axis=None):
        raise schema_error(
            artifact, path, found="rows with missing fields", expected="one field per column in every row"
        )

    validate_text(frame, artifact, path)
    return frame
