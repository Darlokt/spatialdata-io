"""Select a processed scan and persist its measured bands with the public CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from spatialdata import read_zarr
from spatialdata.transformations import get_transformation

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

    def test_masks_on_primary_scans_with_one_cropped_region(self, tmp_path: Path, runner: CliRunner) -> None:
        for region in ("A", "B"):
            write_qptiff(tmp_path / f"{region}.qptiff", stage_position=(1, 2))
        full = np.full((32, 48), 65533, dtype=np.uint16)
        crop = np.arange(80, dtype=np.int32).reshape(8, 10)
        tifffile.imwrite(tmp_path / "full.tif", full)
        tifffile.imwrite(tmp_path / "crop.tif", crop)
        output = tmp_path / "segmented.zarr"
        result = runner.invoke(
            cli,
            [
                "phenocycler",
                "-i",
                str(tmp_path),
                "-o",
                str(output),
                "--labels",
                json.dumps({"A": "full.tif", "B": "crop.tif"}),
                "--labels-to-image",
                '{"B":[[2,0,8],[0,2,6],[0,0,1]]}',
            ],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        stored = read_zarr(output)
        assert set(stored.labels) == {"A_labels", "B_labels"}
        np.testing.assert_array_equal(stored.labels["A_labels"].values, full)
        np.testing.assert_array_equal(stored.labels["B_labels"].values, crop)
        t = get_transformation(stored.labels["B_labels"], "global")
        assert not isinstance(t, dict)
        np.testing.assert_allclose(
            t.to_affine_matrix(("x", "y"), ("x", "y")), [[1, 0, 10004], [0, 1, 20003], [0, 0, 1]]
        )
