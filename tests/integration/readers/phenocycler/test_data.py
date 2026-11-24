"""Read and persist the inForm component image distributed by Akoya."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from spatialdata import read_zarr
from spatialdata.transformations import BaseTransformation, get_transformation

from spatialdata_io._cli import cli

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from click.testing import CliRunner


@pytest.mark.cli
class TestPhenoCyclerDataset:
    def test_akoya_inform_component_image(
        self, require_test_dataset: Callable[[str], Path], tmp_path: Path, runner: CliRunner
    ) -> None:
        root = require_test_dataset("phenocycler_inform")
        source = root / "FIHC4__0929309_HP_IM3_2_component_data.tif"
        output = tmp_path / "components.zarr"
        result = runner.invoke(cli, ["phenocycler", "-i", str(source), "-o", str(output)])
        assert result.exit_code == 0, (result.output, result.exception)

        stored = read_zarr(output)
        assert stored.attrs["spatialdata_io_reader"] == "phenocycler"
        assert len(stored.images) == 1
        image = next(iter(stored.images.values()))
        assert image.dims == ("c", "y", "x")
        assert image.shape == (5, 260, 348)
        assert image.dtype == np.float32
        assert image.coords["c"].values.tolist() == ["Alexa 594", "Cy3", "Cy5", "DAPI", "FITC"]
        with tifffile.TiffFile(source) as tiff:
            for channel in range(5):
                expected = tiff.pages[channel].asarray()
                np.testing.assert_array_equal(image.data[channel].compute(), expected)
        # The fixture's known resolution is 4294967295/213523 pixels/cm on both axes.
        pixel_um = 10000 * 213523 / 4294967295
        transform = get_transformation(image, to_coordinate_system="global")
        assert isinstance(transform, BaseTransformation)
        np.testing.assert_allclose(transform.to_affine_matrix(("x", "y"), ("x", "y")), np.diag([pixel_um, pixel_um, 1]))
