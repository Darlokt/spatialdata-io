"""Small five-artifact Curio export with nonempty feature enrichment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:
    from pathlib import Path


def write_curio(path: Path) -> None:
    """Write sample-prefixed H5AD and four text enrichments."""
    coordinates = np.asarray([[30.0, 4.0], [10.0, 8.0], [20.0, 2.0]])
    table = ad.AnnData(
        X=sparse.csr_array(np.asarray([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.int32)),
        obs=pd.DataFrame({"quality": [3, 2, 1]}, index=["bead-b", "bead-a", "bead-c"]),
        var=pd.DataFrame({"symbol": ["B", "A", "C"]}, index=["gene-b", "gene-a", "gene-c"]),
        layers={"other": np.ones((3, 3), dtype=np.float32)},
    )
    table.obsm["spatial"] = coordinates
    table.obsm["X_spatial"] = coordinates.copy()
    table.write_h5ad(path / "sample_anndata.h5ad")
    (path / "sample_cluster_assignment.txt").write_text("bead-b\t1\nbead-a\t0\nbead-c\t1\n", encoding="utf-8")
    (path / "sample_Metrics.csv").write_text(
        'Metric,Value,Category\n"Reads, total",100,Sequencing\nBeads,3,Mapping\n', encoding="utf-8"
    )
    (path / "sample_variable_features_clusters.txt").write_text(
        "gene\tcluster\tavg_log2FC\ngene-c\t1\t2.5\n", encoding="utf-8"
    )
    (path / "sample_variable_features_spatial_moransi.txt").write_text(
        "\tMoransI_observed\tMoransI_p.value\ngene-c\t0.8\t0.01\ngene-b\t0.4\t0.05\n", encoding="utf-8"
    )
