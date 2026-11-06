"""Discover one Curio Seeker sample export: five sample-prefixed files in one directory.

The Seeker Primary Analysis Pipeline writes one output directory per sample::

    <sample>/
    ├── <sample>_anndata.h5ad                           # counts, bead XY pixels, annotations
    ├── <sample>_cluster_assignment.txt                 # headerless TSV: bead ID, cluster label
    ├── <sample>_Metrics.csv                            # metric, value, ..., Category
    ├── <sample>_variable_features_clusters.txt         # TSV of cluster-defining features
    └── <sample>_variable_features_spatial_moransi.txt  # TSV: feature ID, Moran's I statistics

Each suffix must match exactly one entry, every match must be a regular file
inside the directory (subject to the root's path policy), and all five must
share one nonempty ``<sample>`` prefix. Other entries, such as reports, images
or backups like ``<sample>_anndata.h5ad.backup``, are ignored. The export
carries no format-version marker, so no version is detected or recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactFile,
    normalize_dataset_root,
    require_metadata_file,
    require_unique_path,
    sorted_paths,
)

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.paths import PathPolicy

# Stable artifact names for error contexts; file names vary with the sample.
COUNTS = "counts"
CLUSTERS = "cluster assignments"
METRICS = "metrics"
CLUSTER_FEATURES = "cluster features"
MORAN_FEATURES = "Moran features"

# Artifact names and file suffixes in ``CurioLayout`` field order.
_ARTIFACTS = (
    (COUNTS, "anndata.h5ad"),
    (CLUSTERS, "cluster_assignment.txt"),
    (METRICS, "Metrics.csv"),
    (CLUSTER_FEATURES, "variable_features_clusters.txt"),
    (MORAN_FEATURES, "variable_features_spatial_moransi.txt"),
)


@dataclass(frozen=True, slots=True)
class CurioLayout:
    """The five validated files of one sample export."""

    counts: ArtifactFile
    clusters: ArtifactFile
    metrics: ArtifactFile
    cluster_features: ArtifactFile
    moran_features: ArtifactFile


def discover_layout(path: str | Path, *, path_policy: PathPolicy = "strict") -> CurioLayout:
    """Resolve exactly one file per suffix, all sharing one sample prefix."""
    context = ReaderErrorContext("curio", "dataset layout")
    root = normalize_dataset_root(path, context=context, path_policy=path_policy)
    candidates = sorted_paths(root.path.iterdir(), context=context, search_path=root.path)

    artifacts: list[ArtifactFile] = []
    prefixes: set[str] = set()
    for artifact, suffix in _ARTIFACTS:
        # The pattern makes a missing or ambiguous file actionable.
        artifact_context = ReaderErrorContext("curio", f"{artifact} (*_{suffix})")
        candidate = require_unique_path(
            (item for item in candidates if item.name.endswith(f"_{suffix}")),
            context=artifact_context,
            search_path=root.path,
        )
        prefixes.add(candidate.name.removesuffix(f"_{suffix}"))
        artifacts.append(require_metadata_file(root, candidate, context=artifact_context))

    if len(prefixes) != 1 or "" in prefixes:
        raise ReaderFormatError(
            context=context,
            path=root.path,
            found=f"sample prefixes {sorted(prefixes)!r}",
            expected="one nonempty sample prefix shared by all five artifacts",
        )

    return CurioLayout(*artifacts)
