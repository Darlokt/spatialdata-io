"""Read one Curio Seeker sample export into a linked bead table and bead circles."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, cast

import pandas as pd
from spatialdata import SpatialData

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table, prepare_anndata_dataframe
from spatialdata_io.readers.curio._geometry import read_shapes
from spatialdata_io.readers.curio._layout import CLUSTER_FEATURES, COUNTS, discover_layout
from spatialdata_io.readers.curio._schemas import DEFINING_FEATURES_KEY, STRING_DTYPE
from spatialdata_io.readers.curio._tables import read_table

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.table import SparseDuplicateSummary


def curio(path: str | Path, *, path_policy: Literal["strict", "relaxed"] = "strict") -> SpatialData:
    """Read a Curio Seeker sample export.

    The Seeker Primary Analysis Pipeline writes five files per sample, all
    named with the same nonempty ``<sample>_`` prefix::

        <sample>/
        ├── <sample>_anndata.h5ad
        ├── <sample>_cluster_assignment.txt
        ├── <sample>_Metrics.csv
        ├── <sample>_variable_features_clusters.txt
        └── <sample>_variable_features_spatial_moransi.txt

    Each suffix must match exactly one file; other files are ignored. The
    export has no version marker, so no pipeline version is detected.

    Parameters
    ----------
    path
        Sample directory containing the five files.
    path_policy
        How symlinks inside the dataset directory are treated. ``"strict"``
        requires every discovered artifact and relative file option to resolve
        inside the directory. ``"relaxed"`` also accepts symlinks to targets
        stored elsewhere, as in git-annex or DVC datasets. Relative paths may
        not leave the directory with ``..`` in either mode, and absolute file
        options may point anywhere. File options keep the name the caller
        gave, and relaxed reads keep each artifact's own name rather than that
        of its link target.

    Returns
    -------
    SpatialData
        One table and one shapes element; no images, labels or points.

        ``tables["table"]`` is the H5AD with its beads and features in source
        order:

        - ``X``, dense or sparse in the H5AD, becomes a finite, canonical
          ``scipy.sparse.csr_array`` with the source dtype. Duplicate sparse
          entries are summed. If an integer sum
          overflows, the matrix becomes float32 when every value stays exact,
          otherwise float64, and the read fails if neither is exact.
        - Other layers, ``raw``, ``varm``, ``obsp``, ``varp``, other
          ``obsm`` and ``uns`` entries and the imported ``obs`` and ``var``
          columns are kept as stored. An imported column name or top-level
          ``uns`` key that a SpatialData table does not accept (``my note``)
          is sanitized with a warning (``my_note``; a name that then
          collides, ignoring letter case, with another one or a reader-owned
          one such as ``cluster`` or ``spatialdata_io_positions`` gets a
          ``_1``, ``_2``, ... suffix). ``uns`` values, their order and nested
          keys stay as stored; SpatialData checks only top-level keys.
        - ``obs["cluster"]``: categorical cluster labels, matched by bead ID,
          with categories in order of first appearance along the rows.
        - ``obs["region"]`` (categorical, always ``"cells"``) and
          ``obs["instance_id"]`` (the bead IDs) link the table to
          ``shapes["cells"]``.
        - One ``var`` column per Moran statistic, matched by feature ID:
          int64 or uint64, float64 or bool. Unlisted features are missing,
          which makes integer and boolean columns nullable and float
          columns NaN. A statistic name that a SpatialData table does not
          accept (``MoransI observed``) is sanitized with a warning
          (``MoransI_observed``; a name that then equals another ``var``
          column ignoring letter case gets a ``_1``, ``_2``, ... suffix).
        - ``uns["Top_cluster_defining_features"]``: the cluster-features file
          as a DataFrame of strings, with empty cells missing.
        - ``uns[<Category>]``: one ``{metric: value}`` dictionary of strings
          per metric category. A category that is not a valid ``uns`` key
          (``Run Metrics``) is sanitized with a warning (``Run_Metrics``), as
          is one equal to a sanitized H5AD ``uns`` key ignoring letter case
          (``QC`` next to ``qc`` becomes ``QC_1``); an exact match still
          fails.
        - ``obsm["spatial_pixels"]``: the source ``X_spatial`` array
          unchanged; ``obsm["spatial"]``: float64 micrometres;
          ``uns["spatialdata_io_positions"]`` describes both.

        ``shapes["cells"]`` has one circle per bead, indexed by bead ID in
        table order, centred at its ``X_spatial`` position in Seeker pixels
        with a radius of half the nominal 10 µm bead diameter (5 / 0.647,
        about 7.73 pixels). ``Scale(0.647)`` maps it to the ``global``
        coordinate system in micrometres.

        Provenance in ``attrs`` records the reader and package version.

    Warns
    -----
    UserWarning
        Once per call if duplicate sparse entries in ``X`` were summed,
        naming the H5AD and any dtype change; once each if H5AD ``obs`` or
        ``var`` column names, top-level H5AD ``uns`` keys, Moran statistic
        names or metric categories were sanitized.

    Raises
    ------
    TypeError
        If ``path_policy`` is not a string.
    ValueError
        If ``path_policy`` is not ``"strict"`` or ``"relaxed"``, or if final
        normalization finds nonfinite or malformed sparse expression.
    FileNotFoundError
        If the directory or one of the five files is missing.
    NotADirectoryError
        If ``path`` is not a directory.
    IsADirectoryError
        If a matched file name is a directory.
    ReaderFormatError
        A ``ValueError`` naming the artifact and path if files are ambiguous
        or, under the strict policy, resolve outside the directory, prefixes
        differ, a text file is not
        UTF-8 or violates its column contract, identifiers do not match the
        H5AD, coordinates are invalid or disagree, or an enrichment would
        overwrite an existing H5AD annotation.
    OSError
        If a file cannot be read; the error names the artifact.

    Notes
    -----
    **Coordinates.** Takara's coordinate-unit FAQ gives approximately
    0.647 µm per Seeker pixel for primary-pipeline output. The calibration
    registers no image. ``X_spatial`` must be a finite real ``(n_obs, 2)`` XY
    array that converts to float64 without loss; an optional
    ``obsm["spatial"]`` in the source must equal it exactly. The uppercase
    ``X_SPATIAL`` profile, Trekker and merged or reprocessed objects are not
    supported.

    **Text files.** Values are kept as exact strings: numeric-looking labels,
    leading zeros, ``NA`` and ``TRUE`` stay literal. Header names must be
    nonempty and unique, every row needs one field per column, and NUL
    characters are rejected.

    - Cluster assignments: headerless, exactly two tab-separated fields, bead
      ID and cluster label, one row per H5AD bead.
    - Metrics: the first two columns are metric name and value under any
      header names, followed by a ``Category`` column; further columns are
      ignored. Metric names are unique within a category.
    - Cluster features: tab-separated with named columns, kept in full.
    - Moran statistics: tab-separated; the first column holds known, unique
      feature IDs and may have an empty header. Every other column is a
      statistic: finite numbers, or booleans spelled ``TRUE``/``FALSE``,
      ``True``/``False`` or ``true``/``false``. Integers within the int64 or
      uint64 range are stored exactly. Other numbers of magnitude ``2**53``
      or more must be exactly representable in float64, and nonzero values
      must not underflow to zero.

    **Ownership.** Everything is read eagerly. Source files are never
    modified.
    """
    layout = discover_layout(path, path_policy=path_policy)
    table = read_table(layout)
    shapes = read_shapes(table, layout.counts)

    # Bead IDs link the table rows to the circles built from the same rows.
    table.obs["region"] = pd.Categorical(["cells"] * table.n_obs)
    table.obs["instance_id"] = pd.Series(table.obs_names, index=table.obs_names, dtype=STRING_DTYPE)

    # read_h5ad returns eager DataFrames; AnnData's annotation also includes backed Dataset2D.
    prepare_anndata_dataframe(cast("pd.DataFrame", table.obs), context=ReaderErrorContext("curio", "observations"))
    # Unlisted features have intentionally missing Moran statistics.
    prepare_anndata_dataframe(
        cast("pd.DataFrame", table.var), context=ReaderErrorContext("curio", "features"), allow_float_nulls=True
    )
    prepare_anndata_dataframe(table.uns[DEFINING_FEATURES_KEY], context=ReaderErrorContext("curio", CLUSTER_FEATURES))

    duplicate_sums: list[SparseDuplicateSummary] = []
    table = parse_linked_table(
        table,
        TableLinkage(("cells",), "region", "instance_id"),
        context=ReaderErrorContext("curio", COUNTS),
        path=layout.counts,
        duplicate_sums=duplicate_sums,
        expected_instance_ids={"cells": shapes.index},
    )

    result = SpatialData(tables={"table": table}, shapes={"cells": shapes})
    set_reader_provenance(result.attrs, reader="curio")
    if duplicate_sums:
        warnings.warn(
            f"curio {layout.counts.name}: " + "; ".join(map(str, duplicate_sums)),
            UserWarning,
            stacklevel=2,
        )
    return result
