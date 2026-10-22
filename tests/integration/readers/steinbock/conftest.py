"""Format-faithful multi-image Steinbock AnnData and OME inputs."""

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
class SteinbockDataset:
    """Both segmentation families and two images with independent expectations."""

    root: Path
    images: dict[str, NDArray[np.float32]]
    deepcell: dict[str, NDArray[np.uint16]]
    ilastik: dict[str, NDArray[np.uint16]]


@pytest.fixture
def steinbock_dataset(tmp_path: Path) -> SteinbockDataset:
    """Write the upstream compound observation-ID and metadata conventions."""
    (tmp_path / "ome").mkdir()
    for kind in ("deepcell", "ilastik"):
        (tmp_path / f"masks_{kind}").mkdir()
    images: dict[str, NDArray[np.float32]] = {}
    deepcell: dict[str, NDArray[np.uint16]] = {}
    ilastik: dict[str, NDArray[np.uint16]] = {}
    for offset, name in enumerate(("sample-b", "sample-a")):
        image = np.arange(2 * 8 * 12, dtype=np.float32).reshape(2, 8, 12) + 1000 * offset
        mask = np.zeros((8, 12), dtype=np.uint16)
        mask[1:3, 2:5] = 2
        mask[4:7, 8:10] = 7
        other_mask = np.roll(mask, 1, axis=0)
        tifffile.imwrite(
            tmp_path / "ome" / f"{name}.ome.tiff",
            image,
            ome=True,
            photometric="minisblack",
            metadata={"axes": "CYX", "Channel": {"Name": ["DNA", "CD3"]}},
        )
        tifffile.imwrite(tmp_path / "masks_deepcell" / f"{name}.tiff", mask, photometric="minisblack")
        tifffile.imwrite(tmp_path / "masks_ilastik" / f"{name}.tiff", other_mask, photometric="minisblack")
        images[name] = image
        deepcell[name] = mask
        ilastik[name] = other_mask
    names = ["sample-b.tiff", "sample-a.tiff", "sample-b.tiff", "sample-a.tiff"]
    ids = [2, 7, 7, 2]
    table = ad.AnnData(
        X=csr_array(np.asarray([[1, 0], [0, 4], [5, 6], [7, 8]], dtype=np.float32)),
        obs=pd.DataFrame(
            {
                "Image": names,
                "image": names,
                "centroid-0": [1.5, 5.0, 5.0, 1.5],
                "centroid-1": [3.0, 8.5, 8.5, 3.0],
            },
            index=[f"Object {instance} in {name}" for name, instance in zip(names, ids, strict=True)],
        ),
        var=pd.DataFrame(
            {"Channel Name": ["DNA", "CD3"], "Final Concentration / Dilution": ["1:100", "1:200"]},
            index=["DNA", "CD3"],
        ),
    )
    # Historical exports used this key; enable only while writing that legacy artifact.
    with ad.settings.override(disallow_forward_slash_in_h5ad=False):
        table.write_h5ad(tmp_path / "cells.h5ad")
    return SteinbockDataset(tmp_path, images, deepcell, ilastik)
