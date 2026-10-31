"""Verify the real DBiT command persists a linked grid and compressed raster."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy import sparse
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.transformations import Identity, get_transformation

from spatialdata_io import __version__
from spatialdata_io._cli import cli
from tests.integration.readers.dbit.test_reader import _ORDERED_NAMES, _write_layout

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner


@pytest.mark.cli
class TestDbitCommandWorkflow:
    """Exercise command parsing, actual reader composition and Zarr persistence."""

    def test_writes_linked_spatialdata_zarr(self, tmp_path: Path, runner: CliRunner) -> None:
        source = tmp_path / "input"
        source.mkdir()
        _write_layout(source)
        pixels = np.arange(3 * 16 * 32, dtype=np.uint16).reshape(3, 16, 32)
        tifffile.imwrite(
            source / "image.tif",
            pixels,
            metadata={"axes": "CYX"},
            photometric="minisblack",
            tile=(16, 16),
            compression="deflate",
        )
        output = tmp_path / "cli.zarr"

        result = runner.invoke(
            cli,
            [
                "dbit",
                "-i",
                str(source),
                "-o",
                str(output),
                "--image-file",
                "image.tif",
                "--dataset-id",
                "cli_grid",
                "--chunks",
                "16",
            ],
        )

        assert result.exit_code == 0, result.output
        restored = read_zarr(output)
        assert list(restored.images) == ["cli_grid_image"]
        assert list(restored.shapes) == ["cli_grid"]
        assert list(restored.tables) == ["table"]
        assert not restored.labels
        assert not restored.points
        assert restored.attrs == {
            "spatialdata_io_reader": "dbit",
            "spatialdata_io_software_version": __version__,
        }
        image = restored.images["cli_grid_image"]
        assert image.dims == ("c", "y", "x")
        assert image.dtype == np.dtype(np.uint16)
        np.testing.assert_array_equal(image.c, [0, 1, 2])
        np.testing.assert_array_equal(image.data.compute(), pixels)
        shapes = restored.shapes["cli_grid"]
        assert shapes.index.tolist() == [0, 1]
        np.testing.assert_allclose(shapes.total_bounds, [2, 2, 30, 30])
        assert get_transformation(image, get_all=True) == {"global": Identity()}
        assert get_transformation(shapes, get_all=True) == {"global": Identity()}
        table = restored.tables["table"]
        assert isinstance(table.obs, pd.DataFrame)
        assert table.obs_names.tolist() == _ORDERED_NAMES
        assert table.var_names.tolist() == ["gene-b", "gene-a"]
        assert table.obs["array_A"].tolist() == [1, 2]
        assert table.obs["array_B"].tolist() == [1, 2]
        assert table.obs["sample"].tolist() == ["cli_grid", "cli_grid"]
        assert table.obs["region"].tolist() == ["cli_grid", "cli_grid"]
        assert isinstance(table.obs["region"].dtype, pd.CategoricalDtype)
        assert get_table_keys(table) == ("cli_grid", "region", "pixel_id")
        np.testing.assert_array_equal(table.obs["pixel_id"], shapes.index)
        np.testing.assert_array_equal(table.obsm["spatial_grid"], [[1, 1], [2, 2]])
        assert table.uns["spatialdata_io_positions"]["coordinates"]["spatial_grid"]["units"] == "grid"
        for expression, expected in ((table.X, [1, 2, 3, 4]), (table.layers["counts"], [2, 4, 6, 8])):
            # AnnData persistence can restore sparse matrices instead of arrays.
            assert isinstance(expression, sparse.csr_array | sparse.csr_matrix)
            assert expression.dtype == np.dtype(np.int32)
            assert expression.has_canonical_format
            np.testing.assert_array_equal(expression.data, expected)
            np.testing.assert_array_equal(expression.indices, [0, 1, 0, 1])
            np.testing.assert_array_equal(expression.indptr, [0, 2, 4])
