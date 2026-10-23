"""Exercise complete MERSCOPE command conversions with real source artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest
from numpy.testing import assert_array_equal
from scipy import sparse
from spatialdata import read_zarr
from spatialdata.models import get_table_keys

from spatialdata_io._cli.commands.merscope import merscope_command
from tests.integration.readers.merscope._fixtures import write_merscope_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner


class TestMerscopeCommand:
    """Verify CLI parsing, VPT selection and output storage together."""

    @pytest.mark.parametrize("profile", ["standard", "cellpose", "watershed"])
    def test_complete_conversion(
        self, runner: CliRunner, tmp_path: Path, profile: Literal["standard", "cellpose", "watershed"]
    ) -> None:
        fixture = write_merscope_dataset(tmp_path / "source", profile=profile, reassigned_transcripts=True)
        output = tmp_path / "output.zarr"
        arguments = [
            "--input",
            str(fixture.root),
            "--output",
            str(output),
            "--z-layers",
            "2",
            "--z-layers",
            "0",
            "--chunks",
            "1",
            "--chunks",
            "16",
            "--chunks",
            "20",
        ]
        if profile != "standard":
            arguments.extend(["--vpt-directory", "vpt"])
        result = runner.invoke(merscope_command, arguments)
        assert result.exit_code == 0, result.output
        restored = read_zarr(output)
        assert list(restored.images) == ["slide_region_z0", "slide_region_z2"]
        table = restored.tables["table"]
        assert isinstance(table.X, sparse.csr_array | sparse.csr_matrix)
        assert isinstance(table.obsm["blank"], sparse.csr_array | sparse.csr_matrix)
        assert_array_equal(table.uns["blank_names"], ["Blank-1"])
        assert get_table_keys(table) == ("slide_region_polygons", "cells_region", "EntityID")
        assert_array_equal(restored.shapes["slide_region_polygons"].geometry.area, [19, 4, 4])
        points = restored.points["slide_region_transcripts"].compute()
        expected = ["1001", "2002", "3003", "-1"] if profile == "standard" else ["9001", "9002", "9003", "-1"]
        assert points["cell_id"].tolist() == expected
        assert "z" not in points.columns
        assert_array_equal(points["z_raw"], [0, 2, 2, 0])

    def test_table_only_conversion_does_not_need_raster_or_geometry_sources(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        fixture = write_merscope_dataset(tmp_path / "source")
        for artifact in (fixture.root / "images").iterdir():
            artifact.unlink()
        (fixture.root / "images").rmdir()
        (fixture.root / "detected_transcripts.csv").unlink()
        (fixture.root / "cell_boundaries.parquet").unlink()
        output = tmp_path / "output.zarr"
        result = runner.invoke(
            merscope_command,
            [
                "--input",
                str(fixture.root),
                "--output",
                str(output),
                "--no-include-images",
                "--no-include-transcripts",
                "--no-include-cell-boundaries",
            ],
        )
        assert result.exit_code == 0, result.output
        restored = read_zarr(output)
        assert "spatialdata_attrs" not in restored.tables["table"].uns
        assert not restored.images
        assert not restored.points
        assert not restored.shapes
