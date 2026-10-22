"""Discover one complete, contained Curio analysis export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactFile,
    DatasetRoot,
    normalize_dataset_root,
    require_metadata_file,
    require_unique_path,
    sorted_paths,
)

if TYPE_CHECKING:
    from pathlib import Path

_SUFFIXES = (
    "anndata.h5ad",
    "cluster_assignment.txt",
    "Metrics.csv",
    "variable_features_clusters.txt",
    "variable_features_spatial_moransi.txt",
)


@dataclass(frozen=True, slots=True)
class _CurioLayout:
    """Hold the validated root and five consistently prefixed export artifact paths."""

    root: DatasetRoot
    counts: ArtifactFile
    clusters: ArtifactFile
    metrics: ArtifactFile
    cluster_features: ArtifactFile
    moran_features: ArtifactFile


def _discover_layout(path: str | Path) -> _CurioLayout:
    """Resolve exactly one contained artifact per suffix for a single sample."""
    context = ReaderErrorContext("curio", "dataset layout")
    root = normalize_dataset_root(path, context=context)
    candidates = sorted_paths(root.iterdir(), context=context, search_path=root)
    artifacts: list[ArtifactFile] = []
    prefixes: set[str] = set()
    for suffix in _SUFFIXES:
        context = ReaderErrorContext("curio", suffix)
        candidate = require_unique_path(
            (item for item in candidates if item.name.endswith(f"_{suffix}")), context=context, search_path=root
        )
        prefixes.add(candidate.name.removesuffix(f"_{suffix}"))
        artifacts.append(require_metadata_file(root, candidate, context=context))
    if len(prefixes) != 1 or "" in prefixes:
        raise ReaderFormatError(
            context=ReaderErrorContext("curio", "dataset layout"),
            path=root,
            found=f"sample prefixes {sorted(prefixes)!r}",
            expected="one nonempty sample prefix shared by all five artifacts",
        )
    return _CurioLayout(root, *artifacts)
