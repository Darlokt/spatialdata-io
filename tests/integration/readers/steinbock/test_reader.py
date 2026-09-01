"""Characterize the current public Steinbock workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import tifffile
from scipy import sparse
from spatialdata.models import get_table_keys

from spatialdata_io import __version__
from spatialdata_io.readers.steinbock import steinbock

if TYPE_CHECKING:
    from pathlib import Path


def _write_layout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image_directory = path / "ome"
    mask_directory = path / "masks_deepcell"
    image_directory.mkdir()
    mask_directory.mkdir()

    image = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    labels = np.asarray(
        [[[0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 2, 2], [0, 0, 0, 2, 2]]],
        dtype=np.uint16,
    )
    tifffile.imwrite(image_directory / "sample.ome.tiff", image, photometric="minisblack")
    tifffile.imwrite(mask_directory / "sample.tiff", labels, photometric="minisblack")

    obs = pd.DataFrame(
        {
            "image": ["sample.tiff", "sample.tiff"],
            "Image": ["sample.tiff", "sample.tiff"],
            "centroid-0": [1.0, 3.0],
            "centroid-1": [1.5, 2.5],
        },
        index=["sample 1", "sample 2"],
    )
    var = pd.DataFrame({"Channel Name": ["A", "B"]}, index=["marker-a", "marker-b"])
    table = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[1, 2], [3, 4]], dtype=np.float32)),
        obs=obs,
        var=var,
    )
    table.write_h5ad(path / "cells.h5ad")
    return image, labels[0]


class TestSteinbock:
    """Characterize sample pairing, lazy rasters, cleanup, and linkage."""

    def test_synthetic_layout_preserves_public_contract(self, tmp_path: Path) -> None:
        image, labels = _write_layout(tmp_path)

        result = steinbock(tmp_path)

        assert list(result.images) == ["sample_image"]
        assert list(result.labels) == ["sample_labels"]
        assert list(result.tables) == ["table"]
        assert not result.points
        assert not result.shapes
        assert result.attrs == {
            "spatialdata_io_reader": "steinbock",
            "spatialdata_io_software_version": __version__,
        }

        image_element = result.images["sample_image"]
        assert image_element.shape == (2, 4, 5)
        assert image_element.dtype == np.dtype(np.uint16)
        np.testing.assert_array_equal(image_element.data[:, 1, 2].compute(), image[:, 1, 2])

        labels_element = result.labels["sample_labels"]
        assert labels_element.shape == (4, 5)
        assert labels_element.dtype == np.dtype(np.uint16)
        np.testing.assert_array_equal(labels_element.data[1:3, 1:4].compute(), labels[1:3, 1:4])

        table = result.tables["table"]
        assert sparse.isspmatrix_csr(table.X)
        np.testing.assert_array_equal(table.X.toarray(), np.asarray([[1, 2], [3, 4]], dtype=np.float32))
        assert table.obs_names.tolist() == ["sample 1", "sample 2"]
        assert table.obs["cell_id"].tolist() == [1, 2]
        assert table.obs["region"].tolist() == ["sample_labels", "sample_labels"]
        np.testing.assert_array_equal(table.obsm["spatial"], np.asarray([[1.0, 1.5], [3.0, 2.5]]))
        assert table.var.columns.tolist() == ["Channel_Name"]
        assert get_table_keys(table) == (["sample_labels"], "region", "cell_id")
