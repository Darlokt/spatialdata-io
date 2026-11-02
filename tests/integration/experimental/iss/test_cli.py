"""Verify the real ISS CLI persists its linked table and both rasters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
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

    from tests.integration.experimental.iss.conftest import IssDataset


@pytest.mark.cli
class TestIssCommandWorkflow:
    """Exercise registered command parsing, reader composition and Zarr writing."""

    def test_writes_linked_spatialdata_zarr(self, iss_dataset: IssDataset, tmp_path: Path, runner: CliRunner) -> None:
        output = tmp_path / "cli.zarr"
        result = runner.invoke(
            cli,
            [
                "iss",
                "-i",
                str(iss_dataset.root),
                "-o",
                str(output),
                "--raw-image-file",
                "raw.tiff",
                "--labels-file",
                "labels.tiff",
                "--table-file",
                "cells.h5ad",
                "--instance-key",
                "segmentation_id",
                "--dataset-id",
                "cli_region",
                "--chunks",
                "16",
            ],
        )

        assert result.exit_code == 0, result.output
        restored = read_zarr(output)
        assert list(restored.images) == ["cli_region_raw_image"]
        assert list(restored.labels) == ["cli_region_labels_image"]
        assert list(restored.tables) == ["table"]
        assert not restored.points
        assert not restored.shapes
        assert restored.attrs == {
            "spatialdata_io_reader": "iss",
            "spatialdata_io_software_version": __version__,
        }
        image = restored.images["cli_region_raw_image"]
        labels = restored.labels["cli_region_labels_image"]
        assert image.dims == ("c", "y", "x")
        assert labels.dims == ("y", "x")
        assert image.dtype == labels.dtype == np.dtype(np.uint16)
        assert image.coords["c"].to_numpy().tolist() == [0, 1]
        np.testing.assert_array_equal(image.data.compute(), iss_dataset.raw)
        np.testing.assert_array_equal(labels.data.compute(), iss_dataset.labels)
        assert get_transformation(image, get_all=True) == {"global": Identity()}
        assert get_transformation(labels, get_all=True) == {"global": Identity()}

        table = restored.tables["table"]
        assert table.obs_names.tolist() == ["7", "2"]
        assert table.var_names.tolist() == ["GeneB", "GeneA", "GeneB-1"]
        assert table.var["feature_id"].tolist() == ["f2", "f1", "f3"]
        assert table.obs["cell_type"].tolist() == ["B", "A"]
        assert table.obs["segmentation_id"].tolist() == [7, 2]
        assert table.obs["region"].tolist() == ["cli_region_labels_image"] * 2
        assert get_table_keys(table) == ("cli_region_labels_image", "region", "segmentation_id")
        assert set(table.obs["segmentation_id"]) <= set(np.unique(iss_dataset.labels))
        np.testing.assert_array_equal(table.obsm["spatial_source"], [[2.0, 4.0], [26.0, 9.0]])
        for expression in (table.X, table.layers["counts"]):
            # AnnData persistence can restore matrices instead of sparse arrays.
            assert isinstance(expression, sparse.csr_array | sparse.csr_matrix)
            assert expression.shape == (2, 3)
            assert expression.dtype == np.dtype(np.float32)
            assert expression.has_canonical_format
            np.testing.assert_array_equal(expression.data, [3, 1, 5, 2])
            np.testing.assert_array_equal(expression.indices, [1, 2, 0, 2])
            np.testing.assert_array_equal(expression.indptr, [0, 2, 4])
