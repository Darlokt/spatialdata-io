"""Characterize the current public MCMICRO WSI workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tifffile
from spatialdata.models import get_table_keys

from spatialdata_io import __version__
from spatialdata_io.readers.mcmicro import mcmicro

if TYPE_CHECKING:
    from pathlib import Path


def _write_layout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    (path / "qc").mkdir()
    (path / "qc" / "params.yml").write_text("workflow:\n  tma: false\n", encoding="utf-8")
    pd.DataFrame({"marker_name": ["CD3", "DAPI"], "channel_number": [1, 2]}).to_csv(path / "markers.csv", index=False)

    image_directory = path / "registration"
    labels_directory = path / "segmentation" / "sample"
    quantification_directory = path / "quantification"
    image_directory.mkdir()
    labels_directory.mkdir(parents=True)
    quantification_directory.mkdir()

    image = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    labels = np.asarray(
        [[[0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 2, 2], [0, 0, 0, 2, 2]]],
        dtype=np.uint16,
    )
    tifffile.imwrite(image_directory / "sample.ome.tif", image, photometric="minisblack")
    tifffile.imwrite(labels_directory / "cells.ome.tif", labels, photometric="minisblack")

    pd.DataFrame(
        {
            "CellID": [1, 2],
            "CD3": [10.0, 20.0],
            "DAPI": [30.0, 40.0],
            "X_centroid": [1.5, 3.5],
            "Y_centroid": [2.0, 1.0],
            "area": [4.0, 5.0],
        }
    ).to_csv(quantification_directory / "sample--seg_cells.csv", index=False)
    return image, labels[0]


class TestMcmicro:
    """Characterize WSI element names, values, markers, and linkage."""

    def test_synthetic_wsi_layout_preserves_public_contract(self, tmp_path: Path) -> None:
        image, labels = _write_layout(tmp_path)

        result = mcmicro(tmp_path)

        assert list(result.images) == ["sample_image"]
        assert list(result.labels) == ["sample_cells"]
        assert list(result.tables) == ["sample--seg_cells"]
        assert not result.points
        assert not result.shapes
        assert result.attrs == {
            "spatialdata_io_reader": "mcmicro",
            "spatialdata_io_software_version": __version__,
        }

        image_element = result.images["sample_image"]
        assert image_element.dims == ("c", "y", "x")
        assert image_element.shape == (2, 4, 5)
        assert image_element.coords["c"].to_numpy().tolist() == ["CD3", "DAPI"]
        np.testing.assert_array_equal(image_element.data[:, 1, 2].compute(), image[:, 1, 2])

        labels_element = result.labels["sample_cells"]
        assert labels_element.dims == ("y", "x")
        assert labels_element.shape == (4, 5)
        np.testing.assert_array_equal(labels_element.data[1:3, 1:4].compute(), labels[1:3, 1:4])

        table = result.tables["sample--seg_cells"]
        np.testing.assert_array_equal(np.asarray(table.X), np.asarray([[10.0, 30.0], [20.0, 40.0]]))
        assert table.obs_names.tolist() == ["0", "1"]
        assert table.var_names.tolist() == ["CD3", "DAPI"]
        assert table.obs["CellID"].tolist() == [1, 2]
        assert table.obs["area"].tolist() == [4.0, 5.0]
        np.testing.assert_array_equal(table.obsm["spatial"], np.asarray([[1.5, 2.0], [3.5, 1.0]]))
        assert get_table_keys(table) == ("sample_cells", "region", "CellID")
