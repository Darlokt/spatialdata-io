"""Downloaded OMAP-10 (MACSiQView 0.x) and OMAP-23 (1.x) exports."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import tifffile
from spatialdata import read_zarr
from spatialdata.models import get_channel_names
from spatialdata.transformations import Scale, get_transformation

from spatialdata_io._cli.commands.macsima import macsima_command
from spatialdata_io.readers.macsima import macsima
from tests.integration.readers.macsima._fixtures import full_resolution

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from click.testing import CliRunner
    from spatialdata import SpatialData
    from xarray import DataArray

_OMAP10_FILES = (
    "C-001_S-000_S_APC_R-01_W-C-1_ROI-01_A-CD15_C-VIMC6.tif",
    "C-001_S-000_S_DAPI_R-01_W-C-1_ROI-01_A-DAPI.tif",
    "C-002_S-000_S_FITC_R-01_W-C-1_ROI-01_A-Bcl2_C-REA872.tif",
    "C-002_S-000_S_PE_R-01_W-C-1_ROI-01_A-CD1c_C-REA694.tif",
)
_OMAP23_FILES = (
    "C-001_S-000_S_APC_R-01_W-D01_ROI-01_A-CD3_C-REA1151.tif",
    "C-001_S-000_S_DAPI_R-01_W-D01_ROI-01_A-DAPI.tif",
    "C-002_S-000_S_PE_R-01_W-D01_ROI-01_A-CD279_C-REA1165.tif",
    "C-004_S-000_S_FITC_R-01_W-D01_ROI-01_A-CD66b_C-REA306.tif",
    "C-015_S-000_B_DAPI_R-01_W-D01_ROI-01_A-DAPI.tif",
)
_EXPECTED = {
    "macsima_omap10": pd.DataFrame(
        {
            "name": ["CD15", "DAPI", "Bcl 2", "CD1c"],
            "cycle": [1, 1, 2, 2],
            "imagetype": ["stain"] * 4,
            "well": ["C-1"] * 4,
            "ROI": [1] * 4,
            "fluorophore": ["APC", "DAPI", "FITC", "PE"],
            "clone": ["VIMC6", pd.NA, "REA872", "REA694"],
            "exposure": [2304.0, 40.0, 96.0, 144.0],
        },
        index=["CD15", "DAPI", "Bcl 2", "CD1c"],
    ),
    "macsima_omap23": pd.DataFrame(
        {
            "name": ["CD3", "DAPI", "CD279", "CD66b", "DAPI_background"],
            "cycle": [1, 1, 2, 4, 15],
            "imagetype": ["stain", "stain", "stain", "stain", "bleach"],
            "well": ["D01"] * 5,
            "ROI": [1] * 5,
            "fluorophore": ["APC", "DAPI", "PE", "FITC", "DAPI"],
            "clone": ["REA1151", pd.NA, "REA1165", "REA306", pd.NA],
            "exposure": [1212.52, 51.0, 322.12, 856.68, 51.0],
        },
        index=["CD3", "DAPI", "CD279", "CD66b", "DAPI_background"],
    ),
}
_FILES = {"macsima_omap10": _OMAP10_FILES, "macsima_omap23": _OMAP23_FILES}
_SOFTWARE = {"macsima_omap10": "0.15.3", "macsima_omap23": "1.1.1"}


def _image(result: SpatialData, name: str) -> DataArray:
    return full_resolution(result.images[name])


class TestMacsimaData:
    """The public exports remain the ground truth of both schema versions."""

    @pytest.mark.parametrize("dataset", ["macsima_omap10", "macsima_omap23"])
    def test_exact_metadata_pixels_and_physical_frame(
        self, dataset: str, require_test_dataset: Callable[[str], Path]
    ) -> None:
        root = require_test_dataset(dataset)

        result = macsima(root)

        name = root.name
        image = _image(result, f"{name}_image")
        expected = np.stack([tifffile.imread(root / file) for file in _FILES[dataset]])
        np.testing.assert_array_equal(image.data.compute(), expected)
        assert get_channel_names(result.images[f"{name}_image"]) == _EXPECTED[dataset].index.tolist()
        transformation = get_transformation(result.images[f"{name}_image"], to_coordinate_system=name)
        assert isinstance(transformation, Scale)
        np.testing.assert_array_equal(transformation.scale, [17.0, 17.0])
        table = result.tables[f"{name}_table"]
        assert isinstance(table.var, pd.DataFrame)
        pd.testing.assert_frame_equal(table.var, _EXPECTED[dataset], check_dtype=False, check_index_type=False)
        assert result.attrs["spatialdata_io_format_version"] == _SOFTWARE[dataset]

    @pytest.mark.parametrize(
        ("dataset", "expected"),
        [
            ("macsima_omap10", ["R1 CD15", "R1 DAPI", "R2 Bcl 2", "R2 CD1c"]),
            ("macsima_omap23", ["R1 CD3", "R1 DAPI", "R2 CD279", "R4 CD66b", "R15 DAPI_background"]),
        ],
    )
    def test_cycle_prefixed_names(
        self, dataset: str, expected: list[str], require_test_dataset: Callable[[str], Path]
    ) -> None:
        root = require_test_dataset(dataset)

        result = macsima(root, include_cycle_in_channel_names=True)

        assert get_channel_names(result.images[f"{root.name}_image"]) == expected

    @pytest.mark.parametrize(
        ("dataset", "excluded", "expected"),
        [
            ("macsima_omap10", (2, 3), ["CD15", "DAPI"]),
            ("macsima_omap23", tuple(range(2, 16)), ["CD3", "DAPI"]),
        ],
    )
    def test_excluded_cycles(
        self,
        dataset: str,
        excluded: tuple[int, ...],
        expected: list[str],
        require_test_dataset: Callable[[str], Path],
    ) -> None:
        root = require_test_dataset(dataset)

        result = macsima(root, exclude_cycles=excluded)

        assert get_channel_names(result.images[f"{root.name}_image"]) == expected

    def test_single_folder_stacks_copies_of_one_roi(
        self, tmp_path: Path, require_test_dataset: Callable[[str], Path]
    ) -> None:
        source = require_test_dataset("macsima_omap10")
        shutil.copytree(source, tmp_path / "run" / "copy_1")
        shutil.copytree(source, tmp_path / "run" / "copy_2")

        result = macsima(tmp_path / "run")

        assert get_channel_names(result.images["run_image"]) == [
            "CD15 (1)",
            "CD15 (2)",
            "DAPI (1)",
            "DAPI (2)",
            "Bcl 2 (1)",
            "Bcl 2 (2)",
            "CD1c (1)",
            "CD1c (2)",
        ]
        assert list(result.tables) == ["run_table"]

    def test_multiple_folders_with_exclusion_and_empty_folder(
        self, tmp_path: Path, require_test_dataset: Callable[[str], Path]
    ) -> None:
        shutil.copytree(require_test_dataset("macsima_omap10"), tmp_path / "OMAP10_small")
        shutil.copytree(require_test_dataset("macsima_omap23"), tmp_path / "OMAP23_small")
        (tmp_path / "empty_folder").mkdir()

        with pytest.warns(UserWarning, match="No TIFF files found in .*empty_folder"):
            both = macsima(tmp_path, parsing_style="processed_multiple_folders")
        one = macsima(
            tmp_path,
            parsing_style="processed_multiple_folders",
            exclude_folder_names=["OMAP10_small", "empty_folder"],
        )

        assert list(both.images) == ["OMAP10_small_image", "OMAP23_small_image"]
        assert list(one.images) == ["OMAP23_small_image"]
        assert list(one.tables) == ["OMAP23_small_table"]

    def test_non_ome_tiff_is_skipped(self, tmp_path: Path, require_test_dataset: Callable[[str], Path]) -> None:
        shutil.copy(require_test_dataset("macsima_omap10") / _OMAP10_FILES[0], tmp_path)
        tifffile.imwrite(tmp_path / "tiff_no_metadata.tiff", np.zeros((77, 94), dtype=np.uint16), metadata=None)

        with pytest.warns(UserWarning, match="without OME metadata"):
            result = macsima(tmp_path)

        assert get_channel_names(result.images[f"{tmp_path.name}_image"]) == ["CD15"]

    def test_renamed_release_file(self, tmp_path: Path, require_test_dataset: Callable[[str], Path]) -> None:
        source = require_test_dataset("macsima_omap23") / _OMAP23_FILES[1]
        shutil.copy(source, tmp_path / "OMAP23_DAPI_01.ome.tif")

        result = macsima(tmp_path)

        np.testing.assert_array_equal(
            _image(result, f"{tmp_path.name}_image").data.compute(), tifffile.imread(source)[np.newaxis]
        )

    @pytest.mark.parametrize("dataset", ["macsima_omap10", "macsima_omap23"])
    def test_cli(
        self, tmp_path: Path, runner: CliRunner, dataset: str, require_test_dataset: Callable[[str], Path]
    ) -> None:
        root = require_test_dataset(dataset)

        outcome = runner.invoke(
            macsima_command,
            ["--input", str(root), "--output", str(tmp_path / "data.zarr"), "--exclude-cycles", "2"],
        )

        assert outcome.exit_code == 0, outcome.output
        assert list(read_zarr(tmp_path / "data.zarr").images) == [f"{root.name}_image"]
