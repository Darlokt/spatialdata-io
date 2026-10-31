"""Actual Stereo-seq command composition through persisted SpatialData."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from click.testing import CliRunner
from scipy.sparse import csr_array, csr_matrix
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.transformations import Scale, get_transformation

from spatialdata_io._cli.commands.stereoseq import stereoseq_command
from tests.integration.readers.stereoseq._fixtures import write_dataset

if TYPE_CHECKING:
    from pathlib import Path


class TestStereoseqCommand:
    """Exercise selection, physical coordinates and exact table linkage together."""

    def test_cells_to_zarr(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path / "run", family="outs", products=("cells",))
        destination = tmp_path / "cells.zarr"
        invoked = CliRunner().invoke(
            stereoseq_command,
            ["--input", str(fixture.root), "--output", str(destination), "--cell-bins", "unadjusted"],
        )
        assert invoked.exit_code == 0, invoked.output
        restored = read_zarr(destination)
        assert set(restored.tables) == {"cells_table"}
        assert set(restored.points) == {"cells_centroids"}
        table = restored.tables["cells_table"]
        # AnnData owns the sparse container returned by the storage round trip.
        assert isinstance(table.X, csr_array | csr_matrix)
        assert table.X.dtype == np.uint64
        np.testing.assert_array_equal(table.X.data, [11, 22])
        np.testing.assert_array_equal(table.X.indices, [0, 1])
        np.testing.assert_array_equal(table.X.indptr, [0, 1, 2])
        assert table.obs_names.tolist() == ["200", "100"]
        assert table.var_names.tolist() == ["z", "a"]
        np.testing.assert_array_equal(table.obsm["spatial"], [[5, 10], [15, 20]])
        np.testing.assert_array_equal(table.obsm["spatial_grid"], [[10, 20], [30, 40]])
        assert get_table_keys(table) == ("cells_centroids", "region", "instance_id")
        points = restored.points["cells_centroids"]
        assert table.obs["instance_id"].tolist() == points.compute().index.tolist()
        transform = get_transformation(points, to_coordinate_system="global")
        assert isinstance(transform, Scale)
        np.testing.assert_array_equal(
            transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
            np.diag([0.5, 0.5, 1]),
        )
