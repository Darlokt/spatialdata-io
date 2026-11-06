"""Verify the real Steinbock CLI persists its linked table and paired rasters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.transformations import Identity, get_transformation

from spatialdata_io import __version__
from spatialdata_io._cli import cli

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner

    from tests.integration.readers.steinbock.conftest import SteinbockDataset


@pytest.mark.cli
class TestSteinbockCommandWorkflow:
    """Exercise registered command parsing, reader composition and Zarr writing."""

    def test_writes_linked_spatialdata_zarr(
        self, steinbock_dataset: SteinbockDataset, tmp_path: Path, runner: CliRunner
    ) -> None:
        output = tmp_path / "cli.zarr"
        result = runner.invoke(
            cli,
            [
                "steinbock",
                "-i",
                str(steinbock_dataset.root),
                "-o",
                str(output),
                "--masks-dir",
                "masks",
                "--table-file",
                str(steinbock_dataset.root / "cells.h5ad"),
                "--chunks",
                "16",
            ],
        )

        assert result.exit_code == 0, result.output
        restored = read_zarr(output)
        assert list(restored.images) == ["sample-a_image", "sample-b_image"]
        assert list(restored.labels) == ["sample-a_labels", "sample-b_labels"]
        assert list(restored.tables) == ["table"]
        assert not restored.points
        assert not restored.shapes
        assert restored.attrs == {
            "spatialdata_io_reader": "steinbock",
            "spatialdata_io_software_version": __version__,
        }
        for sample in ("sample-a", "sample-b"):
            image = restored.images[f"{sample}_image"]
            labels = restored.labels[f"{sample}_labels"]
            assert image.dims == ("c", "y", "x")
            assert labels.dims == ("y", "x")
            assert image.dtype == np.dtype(np.float32)
            assert labels.dtype == np.dtype(np.uint16)
            assert image.coords["c"].to_numpy().tolist() == ["DNA", "CD3"]
            np.testing.assert_array_equal(image.data.compute(), steinbock_dataset.images[sample])
            np.testing.assert_array_equal(labels.data.compute(), steinbock_dataset.masks[sample])
            assert get_transformation(image, get_all=True) == {f"{sample}_pixels": Identity()}
            assert get_transformation(labels, get_all=True) == {f"{sample}_pixels": Identity()}

        table = restored.tables["table"]
        assert isinstance(table.obs, pd.DataFrame)
        assert table.obs_names.tolist() == [
            "Object 2 in sample-b.tiff",
            "Object 7 in sample-a.tiff",
            "Object 7 in sample-b.tiff",
            "Object 2 in sample-a.tiff",
        ]
        assert table.var_names.tolist() == ["DNA", "CD3"]
        assert table.var.columns.tolist() == ["Channel_Name", "Final_Concentration"]
        assert table.var["Final_Concentration"].tolist() == ["1:100", "1:200"]
        assert "Image" not in table.obs
        assert table.obs["image"].tolist() == ["sample-b.tiff", "sample-a.tiff"] * 2
        assert table.obs["cell_id"].tolist() == [2, 7, 7, 2]
        assert table.obs["region"].tolist() == ["sample-b_labels", "sample-a_labels"] * 2
        assert get_table_keys(table) == (["sample-b_labels", "sample-a_labels"], "region", "cell_id")
        for sample in ("sample-a", "sample-b"):
            ids = table.obs.loc[table.obs["region"] == f"{sample}_labels", "cell_id"]
            assert set(ids) <= set(np.unique(steinbock_dataset.masks[sample]))
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], [[3, 1.5], [8.5, 5], [8.5, 5], [3, 1.5]])
        assert "spatial" not in table.obsm
        # AnnData persistence can restore matrices instead of sparse arrays.
        assert isinstance(table.X, sparse.csr_array | sparse.csr_matrix)
        assert table.X.dtype == np.dtype(np.float32)
        assert table.X.has_canonical_format
        np.testing.assert_array_equal(table.X.data, [1, 4, 5, 6, 7, 8])
        np.testing.assert_array_equal(table.X.indices, [0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(table.X.indptr, [0, 1, 2, 4, 6])
