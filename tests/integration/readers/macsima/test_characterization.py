"""Preserve the valid legacy MACSima values on synthetic v0 and v1 exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from spatialdata.models import get_channel_names
from spatialdata.transformations import Scale, get_transformation

from spatialdata_io.readers.macsima import macsima
from tests.integration.readers.macsima._fixtures import (
    V0_CHANNELS,
    V1_CHANNELS,
    ChannelSpec,
    full_resolution,
    write_stack,
)

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata import SpatialData
    from xarray import DataArray

_COLUMNS = ["name", "cycle", "imagetype", "well", "ROI", "fluorophore", "clone", "exposure"]


def _image(result: SpatialData, name: str) -> DataArray:
    return full_resolution(result.images[name])


def _scale(result: SpatialData, name: str, frame: str) -> Scale:
    transformation = get_transformation(result.images[name], to_coordinate_system=frame)
    assert isinstance(transformation, Scale)
    return transformation


class TestMacsimaCharacterization:
    """Values the legacy reader produced correctly stay exactly unchanged."""

    def test_v0_values(self, tmp_path: Path) -> None:
        pixels = write_stack(tmp_path / "run_v0", V0_CHANNELS, version=0)

        result = macsima(tmp_path / "run_v0")

        assert list(result.images) == ["run_v0_image"]
        assert list(result.tables) == ["run_v0_table"]
        image = _image(result, "run_v0_image")
        assert image.dims == ("c", "y", "x")
        assert image.dtype == np.dtype(np.uint16)
        assert get_channel_names(result.images["run_v0_image"]) == ["CD15", "DAPI", "Bcl 2", "CD1c"]
        np.testing.assert_array_equal(image.data.compute(), np.stack(pixels))
        np.testing.assert_allclose(_scale(result, "run_v0_image", "global").scale, [0.17, 0.17])

        expected = pd.DataFrame(
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
        )
        table = result.tables["run_v0_table"]
        assert table.n_obs == 0
        assert table.X is None
        assert isinstance(table.var, pd.DataFrame)
        pd.testing.assert_frame_equal(table.var[_COLUMNS], expected, check_dtype=False)

    def test_v1_values_and_bleach_background(self, tmp_path: Path) -> None:
        pixels = write_stack(tmp_path / "run", V1_CHANNELS, version=1)

        result = macsima(tmp_path / "run")

        names = ["CD3", "DAPI", "CD279", "CD66b", "DAPI_background"]
        assert get_channel_names(result.images["run_image"]) == names
        np.testing.assert_array_equal(_image(result, "run_image").data.compute(), np.stack(pixels))
        var = result.tables["run_table"].var
        assert var.index.tolist() == names
        assert var["cycle"].tolist() == [1, 1, 2, 4, 15]
        assert var["imagetype"].tolist() == ["stain", "stain", "stain", "stain", "bleach"]
        assert var["ROI"].tolist() == [1] * 5
        assert var["exposure"].tolist() == [1212.52, 51.0, 322.12, 856.68, 51.0]
        assert var["clone"].isna().tolist() == [False, True, False, False, True]

    def test_channels_are_ordered_by_cycle_then_marker(self, tmp_path: Path) -> None:
        channels = (
            ChannelSpec("Zeta", 2, "PE"),
            ChannelSpec("Beta", 1, "APC"),
            ChannelSpec("Alpha", 2, "FITC"),
        )
        write_stack(tmp_path / "run", channels, version=1)

        result = macsima(tmp_path / "run")

        assert get_channel_names(result.images["run_image"]) == ["Beta", "Alpha", "Zeta"]

    def test_cycle_prefixed_names(self, tmp_path: Path) -> None:
        write_stack(tmp_path / "run", V1_CHANNELS, version=1)

        result = macsima(tmp_path / "run", include_cycle_in_channel_names=True)

        assert get_channel_names(result.images["run_image"]) == [
            "R1 CD3",
            "R1 DAPI",
            "R2 CD279",
            "R4 CD66b",
            "R15 DAPI_background",
        ]

    def test_duplicate_markers_are_numbered(self, tmp_path: Path) -> None:
        channels = (ChannelSpec("DAPI", 1, "DAPI"), ChannelSpec("CD3", 1, "APC"), ChannelSpec("DAPI", 2, "DAPI"))
        write_stack(tmp_path / "run", channels, version=1)

        result = macsima(tmp_path / "run")

        assert get_channel_names(result.images["run_image"]) == ["CD3", "DAPI (1)", "DAPI (2)"]

    def test_nuclei_split_keeps_first_and_last_nuclei_channel(self, tmp_path: Path) -> None:
        channels = (
            ChannelSpec("CD3", 1, "APC"),
            *(ChannelSpec("DAPI", cycle, "DAPI") for cycle in (1, 3, 5, 7)),
        )
        pixels = write_stack(tmp_path / "run", channels, version=1)

        result = macsima(tmp_path / "run")

        main = _image(result, "run_image").data.compute()
        nuclei = _image(result, "run_nuclei_image").data.compute()
        np.testing.assert_array_equal(main, np.stack([pixels[0], pixels[1], pixels[4]]))
        np.testing.assert_array_equal(nuclei, np.stack(pixels[1:]))
        assert result.tables["run_table"].var["cycle"].tolist() == [1, 1, 7]
        assert result.tables["run_nuclei_table"].var["cycle"].tolist() == [1, 3, 5, 7]

    def test_unequal_shapes_are_zero_padded_right_and_bottom(self, tmp_path: Path) -> None:
        channels = (ChannelSpec("CD3", 1, "APC", shape=(10, 12)), ChannelSpec("CD4", 2, "PE", shape=(8, 14)))
        pixels = write_stack(tmp_path / "run", channels, version=1)

        with pytest.warns(UserWarning, match=r"common canvas \(10, 14\)"):
            result = macsima(tmp_path / "run")

        expected = np.zeros((2, 10, 14), dtype=np.uint16)
        expected[0, :10, :12] = pixels[0]
        expected[1, :8, :14] = pixels[1]
        np.testing.assert_array_equal(_image(result, "run_image").data.compute(), expected)

    def test_excluded_cycles_are_omitted(self, tmp_path: Path) -> None:
        write_stack(tmp_path / "run", V1_CHANNELS, version=1)

        result = macsima(tmp_path / "run", exclude_cycles=(2, 4, 15))

        assert get_channel_names(result.images["run_image"]) == ["CD3", "DAPI"]
