"""Actual CLI composition through reader validation and persisted SpatialData."""

from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array, csr_matrix
from spatialdata import read_zarr

from spatialdata_io._cli.commands.visium_hd import visium_hd_command
from tests.integration.readers.visium_hd._fixtures import write_bundle

if TYPE_CHECKING:
    from pathlib import Path


class TestVisiumHDCommand:
    """Exercise the real command, reader and Zarr writer together."""

    def test_minimal_cli_to_zarr(self, tmp_path: Path) -> None:
        bundle = write_bundle(tmp_path / "outs", sizes=(2, 8), segmented=False)
        output = tmp_path / "result.zarr"
        invoked = CliRunner().invoke(
            visium_hd_command,
            [
                "--input",
                str(bundle.root),
                "--output",
                str(output),
                "--bin-sizes",
                "8",
                "--no-include-images",
            ],
        )
        assert invoked.exit_code == 0, invoked.output
        restored = read_zarr(output)
        assert set(restored.shapes) == {"square_008um"}
        assert set(restored.tables) == {"square_008um_table"}
        table = restored.tables["square_008um_table"]
        # AnnData controls the sparse container returned by storage round trips.
        assert isinstance(table.X, csr_array | csr_matrix)
        expected = csr_array(bundle.raw[8][bundle.filtered_rows[8]])
        assert_array_equal(table.X.data, expected.data)
        assert_array_equal(table.X.indices, expected.indices)
        assert_array_equal(table.X.indptr, expected.indptr)
        assert table.var_names.tolist() == ["Gene", "Gene-1", "Other"]
        assert table.obs["instance_id"].tolist() == restored.shapes["square_008um"].index.tolist()
        assert restored.coordinate_systems == ["global"]

    def test_rejects_image_options_when_images_disabled(self, tmp_path: Path) -> None:
        bundle = write_bundle(tmp_path / "outs", sizes=(2,), segmented=False)
        output = tmp_path / "result.zarr"
        invoked = CliRunner().invoke(
            visium_hd_command,
            [
                "--input",
                str(bundle.root),
                "--output",
                str(output),
                "--no-include-images",
                "--data-axes",
                "y",
                "--data-axes",
                "x",
            ],
        )
        assert invoked.exit_code == 1
        assert isinstance(invoked.exception, ValueError)
        assert "require include_images=True" in str(invoked.exception)
        assert not output.exists()
