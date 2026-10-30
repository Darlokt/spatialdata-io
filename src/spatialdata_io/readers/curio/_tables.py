"""Own H5AD loading and identifier-aligned Curio enrichment."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, cast

import anndata as ad
import pandas as pd

from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers.curio._schemas import (
    DEFINING_FEATURES_KEY,
    STRING_DTYPE,
    normalize_metadata,
    schema_error,
    validate_clusters,
    validate_counts,
    validate_header,
    validate_metrics,
    validate_moran,
)

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.curio._layout import _CurioLayout


def _read_named_table(path: ArtifactFile, *, delimiter: str, feature_ids: bool = False) -> pd.DataFrame:
    """Read the raw header before parsing lexical Arrow annotation fields."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        header = source.readline()

        columns = next(csv.reader([header], delimiter=delimiter, strict=True), [])
        validate_header(columns, path, feature_ids=feature_ids)

        # pandas' Python engine preserves NUL and distinguishes missing fields;
        # its index_col=[] path needs an explicit header-only case.
        position = source.tell()
        if not source.read(1):
            return pd.DataFrame({column: pd.Series([], dtype=STRING_DTYPE) for column in columns})
        source.seek(position)

        try:
            frame = pd.read_csv(
                source,
                sep=delimiter,
                names=columns,
                header=None,
                dtype=STRING_DTYPE,
                keep_default_na=False,
                engine="python",
                index_col=[],
            )
        except pd.errors.ParserError as error:
            raise schema_error(
                path, found=f"malformed enrichment rows: {error}", expected="one field per enrichment header column"
            ) from error

    if not isinstance(frame.index, pd.RangeIndex) or frame.isna().any(axis=None):
        raise schema_error(path, found="inconsistent row width", expected="one field per enrichment header column")

    return frame


def _read_table(layout: _CurioLayout) -> ad.AnnData:
    """Load owned counts and align enrichments without changing expression order."""
    try:
        table = ad.read_h5ad(layout.counts)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=ReaderErrorContext("curio", "counts"), path=layout.counts)
        raise

    validate_counts(table, layout.counts)

    try:
        clusters = pd.read_csv(
            layout.clusters, sep="\t", header=None, dtype=STRING_DTYPE, keep_default_na=False, engine="python"
        )

        validate_clusters(clusters, table.obs_names, layout.clusters)
        clusters.columns = ["bead_id", "cluster"]
        normalize_metadata(clusters, layout.clusters, strings=("bead_id",))
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        add_reader_context(error, context=ReaderErrorContext("curio", "cluster assignments"), path=layout.clusters)
        raise

    table.obs["cluster"] = pd.Categorical(clusters.set_index("bead_id")["cluster"].reindex(table.obs_names))

    _enrich_metrics(table, layout.metrics)

    try:
        defining = _read_named_table(layout.cluster_features, delimiter="\t")
        defining = defining.replace("", pd.NA)
        table.uns[DEFINING_FEATURES_KEY] = normalize_metadata(defining, layout.cluster_features, strings=("gene",))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as error:
        add_reader_context(error, context=ReaderErrorContext("curio", "cluster features"), path=layout.cluster_features)
        raise

    # read_h5ad returns eager DataFrames; AnnData's annotation also includes backed Dataset2D.
    table.var = _enrich_features(cast("pd.DataFrame", table.var), layout.moran_features)
    return table


def _enrich_metrics(table: ad.AnnData, path: ArtifactFile) -> None:
    """Attach validated metric dictionaries with source category order."""
    try:
        metrics = _read_named_table(path, delimiter=",")
        validate_metrics(metrics, path, set(table.uns))

        normalize_metadata(metrics, path, strings=(metrics.columns[0], "Category"))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as error:
        add_reader_context(error, context=ReaderErrorContext("curio", "metrics"), path=path)
        raise

    for category, group in metrics.groupby("Category", sort=False):
        table.uns[str(category)] = dict(zip(group.iloc[:, 0], group.iloc[:, 1], strict=True))


def _enrich_features(features: pd.DataFrame, path: ArtifactFile) -> pd.DataFrame:
    """Join finite statistics by feature ID without changing count-column order."""
    try:
        moran = _read_named_table(path, delimiter="\t", feature_ids=True)
        identifier = moran.columns[0]

        normalize_metadata(moran, path, strings=(identifier,))
        moran = moran.set_index(identifier)
        validate_moran(moran, features, path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as error:
        add_reader_context(error, context=ReaderErrorContext("curio", "Moran features"), path=path)
        raise

    # Match the existing H5AD index representation so pandas preserves it through the join.
    moran.index = moran.index.astype(features.index.dtype)
    return features.join(moran, how="left", validate="one_to_one")
