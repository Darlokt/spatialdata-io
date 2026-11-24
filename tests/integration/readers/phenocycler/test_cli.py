"""Select a processed scan and persist its measured bands with the public CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from spatialdata import read_zarr

from spatialdata_io._cli import cli
from tests.integration.readers.phenocycler._fixtures import write_qptiff

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner


@pytest.mark.cli
class TestPhenoCyclerCommandWorkflow:
    def test_selected_scan_with_display_names(self, tmp_path: Path, runner: CliRunner) -> None:
        scans = tmp_path / "scans"
        scans.mkdir()
        expected = write_qptiff(scans / "Tonsil_Scan1.qptiff")
        write_qptiff(scans / "Thymus_Scan1.qptiff")
        output = tmp_path / "selected.zarr"

        result = runner.invoke(
            cli,
            [
                "phenocycler",
                "-i",
                str(scans),
                "-o",
                str(output),
                "--regions",
                "Tonsil",
                "--channel-names",
                "Nuclei",
                "--channel-names",
                "T cells",
            ],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        stored = read_zarr(output)
        assert list(stored.images) == ["Tonsil_image"]
        image = stored.images["Tonsil_image"]
        assert image.coords["c"].values.tolist() == ["Nuclei", "T cells"]
        np.testing.assert_array_equal(image.data.compute(), expected)
