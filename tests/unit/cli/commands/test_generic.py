"""Test the Generic command boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata_io._cli.commands.generic import generic_command

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestGenericCommand:
    """Verify Click conversion to the Generic converter contract."""

    def test_forwards_options_and_reports_new_output(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture
    ) -> None:
        converter = mocker.patch("spatialdata_io.converters.generic_to_zarr.generic_to_zarr")
        image = tmp_path / "image.tif"
        image.touch()
        output = tmp_path / "output.zarr"

        result = runner.invoke(
            generic_command,
            [
                "--input",
                str(image),
                "--output",
                str(output),
                "--name",
                "raster",
                "--data-axes",
                "cyx",
                "--tiff-series",
                "2",
                "--tiff-level",
                "1",
                "--chunks",
                "1",
                "--chunks",
                "8",
                "--chunks",
                "9",
                "--coordinate-system",
                "cells",
            ],
        )

        assert result.exit_code == 0, result.output
        converter.assert_called_once_with(
            path=image,
            output=output,
            name="raster",
            data_axes="cyx",
            coordinate_system="cells",
            tiff_series=2,
            tiff_level=1,
            chunks=(1, 8, 9),
        )
        assert result.output == f"Data written to {output}\n"
