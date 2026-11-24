"""Persist a native Processor marker stack through the public command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from spatialdata import read_zarr

from spatialdata_io._cli import cli

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner


@pytest.mark.cli
class TestCodexCommandWorkflow:
    def test_selects_one_native_region(self, tmp_path: Path, runner: CliRunner) -> None:
        root = tmp_path / "processed_2020-01-01"
        plane = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48)
        for region in ["reg001", "reg002"]:
            folder = root / "stitched" / region
            folder.mkdir(parents=True)
            tifffile.imwrite(folder / f"{region}_cyc010_ch002_CD3.tif", plane + 7)
            tifffile.imwrite(folder / f"{region}_cyc002_cha001_DAPI.tif", plane)

        output = tmp_path / "out.zarr"
        result = runner.invoke(
            cli, ["codex", "-i", str(root), "-o", str(output), "--regions", "reg002", "--no-read-tables"]
        )
        assert result.exit_code == 0, (result.output, result.exception)
        stored = read_zarr(output)
        assert list(stored.images) == ["reg002_image"]
        assert not stored.tables
        image = stored.images["reg002_image"]
        assert image.coords["c"].values.tolist() == ["DAPI", "CD3"]
        assert stored.attrs["akoya"]["regions"]["reg002"]["channels"] == [
            {"biomarker": "DAPI", "cycle": 2, "channel": 1},
            {"biomarker": "CD3", "cycle": 10, "channel": 2},
        ]
        np.testing.assert_array_equal(image.data.compute(), np.stack([plane, plane + 7]))
