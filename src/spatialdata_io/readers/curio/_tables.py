"""Own H5AD loading and identifier-aligned Curio enrichment."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.curio._layout import _CurioLayout

_MAX_ENRICHMENT_HEADER_CHARACTERS = 1_048_576
_CLUSTER_COLUMNS = 2
_METRIC_COLUMNS = 3


def _format_error(path: ArtifactFile, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a malformed table artifact with its validated source path."""
    return ReaderFormatError(context=ReaderErrorContext("curio", path.name), path=path, found=found, expected=expected)


def _read_table(layout: _CurioLayout) -> ad.AnnData:
    """Load owned counts and align export enrichment without reordering observations or features."""
    try:
        table = ad.read_h5ad(layout.counts)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=ReaderErrorContext("curio", "counts"), path=layout.counts)
        raise

    if table.X is None:
        raise _format_error(
            layout.counts,
            found="missing expression in AnnData.X",
            expected="an observation-by-feature expression array in AnnData.X",
        )
    if table.n_obs == 0 or not table.obs_names.is_unique or not table.var_names.is_unique:
        raise _format_error(
            layout.counts,
            found="empty observations or duplicate identifiers",
            expected="nonempty unique observations and unique features",
        )

    # Cluster rows may be reordered. Observation order belongs to the counts.
    try:
        clusters = pd.read_csv(layout.clusters, sep="\t", header=None, dtype={0: str}, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        add_reader_context(error, context=ReaderErrorContext("curio", "cluster assignments"), path=layout.clusters)
        raise
    if clusters.shape[1] != _CLUSTER_COLUMNS or bool(clusters.isin([""]).any(axis=None)):
        raise _format_error(
            layout.clusters, found="malformed cluster assignments", expected="two non-null columns: bead ID and cluster"
        )
    cluster_ids = pd.Index(clusters.iloc[:, 0])
    if not cluster_ids.is_unique or set(cluster_ids) != set(table.obs_names):
        raise _format_error(
            layout.clusters,
            found="duplicate or mismatched bead IDs",
            expected="exactly one cluster assignment per H5AD observation",
        )
    table.obs["cluster"] = pd.Categorical(clusters.set_index(clusters.columns[0]).iloc[:, 0].reindex(table.obs_names))

    _enrich_metrics(table, layout.metrics)
    try:
        table.uns["Top_cluster_defining_features"] = pd.read_csv(layout.cluster_features, sep="\t")
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        add_reader_context(error, context=ReaderErrorContext("curio", "cluster features"), path=layout.cluster_features)
        raise

    # read_h5ad returns an eager DataFrame. AnnData also annotates backed Dataset2D.
    table.var = _enrich_features(cast("pd.DataFrame", table.var), layout.moran_features)
    return table


def _enrich_metrics(table: ad.AnnData, path: ArtifactFile) -> None:
    """Attach metric dictionaries without pandas header or row-width repair."""
    # Inspect original names before pandas can repair duplicate headers or infer an index.
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            header = source.readline(_MAX_ENRICHMENT_HEADER_CHARACTERS + 1)
            columns = next(csv.reader([header], strict=True))
            if (
                len(header) > _MAX_ENRICHMENT_HEADER_CHARACTERS
                or not columns
                or any(not column for column in columns)
                or len(set(columns)) != len(columns)
            ):
                raise _format_error(
                    path, found="invalid metric header", expected="bounded nonempty unique column names"
                )
            metrics = pd.read_csv(
                source, names=columns, header=None, dtype={0: str, "Category": str}, keep_default_na=False
            )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as error:
        add_reader_context(error, context=ReaderErrorContext("curio", "metrics"), path=path)
        raise

    if (
        not isinstance(metrics.index, pd.RangeIndex)
        or len(metrics.columns) < _METRIC_COLUMNS
        or "Category" not in metrics.columns[2:]
    ):
        raise _format_error(
            path,
            found=f"columns {metrics.columns.tolist()!r}, inferred row index {type(metrics.index).__name__}",
            expected="metric and value columns followed by Category, with matching row widths",
        )

    metric_key = metrics.columns[0]
    if (
        bool(metrics[[metric_key, "Category"]].isin([""]).any(axis=None))
        or metrics.duplicated(["Category", metric_key]).any()
    ):
        raise _format_error(
            path,
            found="empty or duplicate metric identifiers",
            expected="nonempty categories and unique metric names within each category",
        )

    for category, group in metrics.groupby("Category", sort=False):
        table.uns[str(category)] = dict(zip(group.iloc[:, 0], group.iloc[:, 1], strict=True))


def _enrich_features(features: pd.DataFrame, path: ArtifactFile) -> pd.DataFrame:
    """Join finite statistics by feature ID without changing count-column order."""
    # R exports may omit the row-name header. Statistic names must remain unambiguous.
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            header = source.readline(_MAX_ENRICHMENT_HEADER_CHARACTERS + 1)
            columns = next(csv.reader([header], delimiter="\t", strict=True))
            if (
                len(header) > _MAX_ENRICHMENT_HEADER_CHARACTERS
                or not columns
                or any(not column for column in columns[1:])
                or len(set(columns)) != len(columns)
            ):
                raise _format_error(
                    path,
                    found="invalid Moran header",
                    expected="bounded unique column names and nonempty statistic names",
                )
            moran = pd.read_csv(
                source, sep="\t", names=columns, header=None, dtype={0: str}, index_col=0, keep_default_na=False
            )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as error:
        add_reader_context(error, context=ReaderErrorContext("curio", "Moran features"), path=path)
        raise

    if moran.shape[1] == 0 or moran.columns.tolist() != columns[1:]:
        raise _format_error(
            path,
            found="missing statistics or inconsistent row width",
            expected="feature IDs in the first column followed by Moran statistics",
        )

    feature_ids = moran.index
    if not feature_ids.is_unique or bool(feature_ids.isin([""]).any()) or not feature_ids.isin(features.index).all():
        raise _format_error(
            path,
            found="duplicate, empty, or unknown feature IDs",
            expected="unique feature IDs contained in H5AD var_names",
        )
    if len(moran.columns.intersection(features.columns)):
        raise _format_error(
            path,
            found="overlapping H5AD variable columns",
            expected="Moran statistic columns distinct from existing variable metadata",
        )
    if len(moran) and any(
        not pd.api.types.is_numeric_dtype(moran[column].dtype) or not np.isfinite(moran[column].to_numpy()).all()
        for column in moran
    ):
        raise _format_error(
            path,
            found="nonnumeric or nonfinite Moran statistics",
            expected="finite numeric or boolean statistics for each listed feature",
        )

    return features.join(moran, how="left", validate="one_to_one")
