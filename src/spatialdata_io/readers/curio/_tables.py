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

Text files are UTF-8, optionally with a byte-order mark. Arrow parses literal
strings with explicit schemas and validates full record widths. Each artifact
owns its eager read; NUL characters are rejected before alignment. The files
are small (one row per bead, metric or listed feature).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anndata as ad
import pandas as pd
import pyarrow.csv as pacsv

from spatialdata_io.readers._utils.csv import csv_errors, csv_header
from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers._utils.names import sanitize_column_names, sanitize_uns_keys
from spatialdata_io.readers.curio._layout import CLUSTER_FEATURES, CLUSTERS, COUNTS, METRICS, MORAN_FEATURES
from spatialdata_io.readers.curio._schemas import (
    CLUSTER_TYPES,
    DEFINING_FEATURES_KEY,
    RESERVED_OBS,
    RESERVED_UNS,
    normalize_moran,
    text_types,
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

    clusters = _read_clusters(layout.clusters)
    validate_clusters(clusters, table.obs_names, layout.clusters)
    labels = clusters.set_index("bead_id")["cluster"].reindex(table.obs_names)
    table.obs["cluster"] = pd.Categorical(labels, categories=labels.unique())

    _add_metrics(table, layout.metrics)

    defining = _read_defining_features(layout.cluster_features)
    table.uns[DEFINING_FEATURES_KEY] = defining

    table.var = _join_moran(var, layout.moran_features)
    return table


def _read_clusters(path: ArtifactFile) -> pd.DataFrame:
    """Read the two positional cluster-assignment fields as literal strings."""
    context = ReaderErrorContext("curio", CLUSTERS)
    try:
        with path.open("rb") as source, csv_errors(context=context, path=path):
            clusters = pacsv.read_csv(
                source,
                read_options=pacsv.ReadOptions(column_names=CLUSTER_TYPES.names, use_threads=False),
                parse_options=pacsv.ParseOptions(delimiter="\t", newlines_in_values=True),
                convert_options=pacsv.ConvertOptions(
                    column_types=CLUSTER_TYPES,
                    include_columns=CLUSTER_TYPES.names,
                    null_values=[],
                    strings_can_be_null=False,
                ),
            ).to_pandas(types_mapper=pd.ArrowDtype)
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise
    validate_text(clusters, CLUSTERS, path)
    return clusters


def _add_metrics(table: ad.AnnData, path: ArtifactFile) -> None:
    """Store each metric category as a ``{metric: value}`` dictionary in source order."""
    context = ReaderErrorContext("curio", METRICS)
    columns = list(csv_header(path, context=context))
    validate_header(columns, METRICS, path)
    try:
        with path.open("rb") as source, csv_errors(context=context, path=path):
            metrics = pacsv.read_csv(
                source,
                read_options=pacsv.ReadOptions(use_threads=False),
                parse_options=pacsv.ParseOptions(newlines_in_values=True),
                convert_options=pacsv.ConvertOptions(
                    column_types=text_types(columns),
                    include_columns=columns,
                    null_values=[],
                    strings_can_be_null=False,
                ),
            ).to_pandas(types_mapper=pd.ArrowDtype)
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise
    validate_text(metrics, METRICS, path)
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


def _read_defining_features(path: ArtifactFile) -> pd.DataFrame:
    """Read every named defining-feature field literally, retaining source column order."""
    context = ReaderErrorContext("curio", CLUSTER_FEATURES)
    columns = list(csv_header(path, context=context, delimiter="\t"))
    validate_header(columns, CLUSTER_FEATURES, path)
    try:
        with path.open("rb") as source, csv_errors(context=context, path=path):
            defining = pacsv.read_csv(
                source,
                read_options=pacsv.ReadOptions(use_threads=False),
                parse_options=pacsv.ParseOptions(delimiter="\t", newlines_in_values=True),
                convert_options=pacsv.ConvertOptions(
                    column_types=text_types(columns),
                    include_columns=columns,
                    null_values=[""],
                    strings_can_be_null=True,
                ),
            ).to_pandas(types_mapper=pd.ArrowDtype)
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise
    validate_text(defining, CLUSTER_FEATURES, path)
    return defining


def _join_moran(features: pd.DataFrame, path: ArtifactFile) -> pd.DataFrame:
    """Join parsed Moran statistics onto ``var`` by feature ID, keeping feature order."""
    context = ReaderErrorContext("curio", MORAN_FEATURES)
    columns = list(csv_header(path, context=context, delimiter="\t"))
    validate_header(columns, MORAN_FEATURES, path, feature_ids=True)
    try:
        with path.open("rb") as source, csv_errors(context=context, path=path):
            moran = pacsv.read_csv(
                source,
                read_options=pacsv.ReadOptions(use_threads=False),
                parse_options=pacsv.ParseOptions(delimiter="\t", newlines_in_values=True),
                convert_options=pacsv.ConvertOptions(
                    column_types=text_types(columns),
                    include_columns=columns,
                    null_values=[],
                    strings_can_be_null=False,
                ),
            ).to_pandas(types_mapper=pd.ArrowDtype)
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise
    validate_text(moran, MORAN_FEATURES, path)
    moran = moran.set_index(moran.columns[0])
    validate_moran(moran, features, path)
    normalize_moran(moran, path)
    moran.columns = sanitize_column_names(
        list(moran.columns), source=f"curio: the {MORAN_FEATURES} header", reserved=list(features.columns)
    )

    # Match the H5AD index representation so the join keeps it.
    moran.index = moran.index.astype(features.index.dtype)
    return features.join(moran, how="left", validate="one_to_one")
