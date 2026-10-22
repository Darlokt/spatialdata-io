"""Deterministic raw-image, labels, and h5ad ISS input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy.sparse import csr_array

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class IssDataset:
    """Three-artifact ISS input and independent expected raster values."""

    root: Path
    raw: NDArray[np.uint16]
    labels: NDArray[np.uint16]


def write_iss_dataset(path: Path, *, channels: int = 2, height: int = 32) -> IssDataset:
    """Write a multi-page CYX image, YX labels, and an owned expression table."""
    raw = np.arange(channels * height * 40, dtype=np.uint16).reshape(channels, height, 40)
    labels = np.zeros((height, 40), dtype=np.uint16)
    labels[:, 1:4] = 7
    labels[:, 24:28] = 2
    tifffile.imwrite(path / "raw.tiff", raw, photometric="minisblack", metadata={"axes": "CYX"})
    tifffile.imwrite(path / "labels.tiff", labels, photometric="minisblack", metadata={"axes": "YX"})
    expression = csr_array(np.asarray([[0, 3, 1], [5, 0, 2]], dtype=np.float32))
    table = ad.AnnData(
        X=expression,
        obs=pd.DataFrame({"segmentation_id": [7, 2], "cell_type": ["B", "A"]}, index=["7", "2"]),
        var=pd.DataFrame({"feature_id": ["f2", "f1", "f3"]}, index=["GeneB", "GeneA", "GeneB"]),
    )
    table.obsm["spatial"] = np.asarray([[2.0, 4.0], [26.0, 9.0]])
    table.layers["counts"] = expression.copy()
    table.write_h5ad(path / "cells.h5ad")
    return IssDataset(path, raw, labels)


@pytest.fixture
def iss_dataset(tmp_path: Path) -> IssDataset:
    """Return a fresh on-disk input with nontrivial identifiers and features."""
    return write_iss_dataset(tmp_path)
