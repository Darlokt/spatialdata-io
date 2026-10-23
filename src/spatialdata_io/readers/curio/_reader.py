"""Assemble the Curio linked table, bead geometry, and provenance."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd
from spatialdata import SpatialData

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table, prepare_anndata_dataframe
from spatialdata_io.readers.curio._geometry import _read_shapes
from spatialdata_io.readers.curio._layout import _discover_layout
from spatialdata_io.readers.curio._schemas import DEFINING_FEATURES_KEY, STRING_DTYPE
from spatialdata_io.readers.curio._tables import _read_table

if TYPE_CHECKING:
    from pathlib import Path


def curio(path: str | Path) -> SpatialData:
    """Read an unversioned Curio Seeker five-artifact analysis export.

    The directory must contain exactly one file per suffix, all with the same
    nonempty ``<sample>_`` prefix::

        <sample>_anndata.h5ad
        <sample>_cluster_assignment.txt
        <sample>_Metrics.csv
        <sample>_variable_features_clusters.txt
        <sample>_variable_features_spatial_moransi.txt

    Additional unrelated outputs are ignored. No pipeline version is inferred.

    Parameters
    ----------
    path
        Directory containing all five artifacts. Symlinked artifacts must remain
        inside the dataset root.

    Returns
    -------
    SpatialData
        ``tables['table']`` contains owned, eager, finite canonical CSR sparse
        array expression with source dtype, observation order, and feature order.
        Other layers retain their source representation. Cluster assignments are
        aligned by bead ID and stored as categorical ``obs['cluster']``.
        Metric categories become dictionaries in ``uns``. Metric and value
        occupy the first two positional columns, whose names may vary;
        ``Category`` is a named trailing column. Extra metric columns are ignored.
        The cluster-feature
        TSV is stored in ``uns['Top_cluster_defining_features']``. The first
        Moran TSV column contains feature IDs. Statistic columns join
        onto ``var`` in feature order, with missing entries for unlisted genes.
        Supplied statistics must be finite numeric or boolean values. Undefined
        source statistics are rejected.
        Headerless cluster files have exactly two fields: lexical bead ID and
        cluster label. Defining-feature fields retain source column order and
        additional annotations; optional ``gene`` values remain lexical strings.
        Duplicate or empty named headers and NUL characters are rejected.
        Imported H5AD annotations retain their source representations. Imported
        ``cluster``, ``region``, ``instance_id``, or
        ``Top_cluster_defining_features`` destinations and conflicting metric
        categories are rejected instead of overwritten.
        Newly read metadata uses Arrow storage during processing, followed by
        one conversion per completed annotation frame for AnnData persistence.
        Complete numeric outputs use NumPy storage; integer and boolean Moran
        columns with unlisted genes use nullable pandas storage, preserving
        exact large integers. Missing floating statistics become NaN. Newly
        created string annotations use pandas nullable strings. Defining-feature
        empty fields become missing values; literal identifier strings such as
        ``NA`` remain strings.
        ``shapes['cells']`` contains points with radius 10 in source coordinate
        units, indexed by observation ID, and an identity transform to
        ``global``. Coordinates are ordered ``(x, y)`` and must be losslessly
        representable as float64 geometry. No physical-unit conversion is inferred.
        These circles represent beads, despite the historical region name.
        ``obsm['spatial']`` is retained and its identical ``X_spatial`` duplicate
        removed. Final linkage uses ``region`` and
        ``instance_id``. No images, labels, or transcript points are returned.
        All I/O is eager. Caller-owned state is not mutated.

    Raises
    ------
    FileNotFoundError
        If the directory or any required artifact is missing.
    NotADirectoryError
        If the root is not a directory.
    IsADirectoryError
        If a required artifact is a directory.
    ValueError
        If expression is missing or malformed, artifacts are ambiguous, prefixes
        conflict, identifiers disagree, enrichment schemas or supplied Moran
        statistics are invalid, or spatial arrays disagree or lose coordinate
        precision. Format failures use the contextual ``ReaderFormatError``
        subclass.
    """
    layout = _discover_layout(path)
    table = _read_table(layout)
    shapes = _read_shapes(table, layout.counts)

    # Bead IDs link to the completed geometry in the counts observation order.
    table.obs["region"] = pd.Categorical(["cells"] * table.n_obs)
    table.obs["instance_id"] = pd.Series(table.obs_names, index=table.obs_names, dtype=STRING_DTYPE)
    # Only completed, reader-owned frames cross the temporary storage boundary.
    # Moran joins introduce intentional missing statistics for unlisted features.
    # These frames are eager; AnnData also annotates the backed Dataset2D alternative.
    prepare_anndata_dataframe(cast("pd.DataFrame", table.obs), context=ReaderErrorContext("curio", "observations"))
    prepare_anndata_dataframe(
        cast("pd.DataFrame", table.var), context=ReaderErrorContext("curio", "features"), allow_float_nulls=True
    )
    prepare_anndata_dataframe(
        table.uns[DEFINING_FEATURES_KEY],
        context=ReaderErrorContext("curio", "cluster features"),
        allow_float_nulls=True,
    )
    table = parse_linked_table(
        table,
        TableLinkage(("cells",), "region", "instance_id"),
        context=ReaderErrorContext("curio", "linked counts"),
        expected_instance_ids={"cells": shapes.index},
    )

    result = SpatialData(tables={"table": table}, shapes={"cells": shapes})
    set_reader_provenance(result.attrs, reader="curio")
    return result
