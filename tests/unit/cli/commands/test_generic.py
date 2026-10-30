"""Test the Generic command boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io._cli.commands.generic import generic_command
from spatialdata_io._raster_defaults import DEFAULT_SCALE_FACTORS

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
                "--scale-factors",
                "2",
                "--scale-factors",
                "4",
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
            scale_factors=(2, 4),
            chunks=(1, 8, 9),
            invalid_geometries="keep",
        )
        assert result.output == f"Data written to {output}\n"

    @pytest.mark.parametrize(("arguments", "expected"), [((), None), (("--chunks", "64"), 64)])
    def test_forwards_reader_defaults_and_one_size_for_every_axis(
        self,
        tmp_path: Path,
        runner: CliRunner,
        mocker: MockerFixture,
        arguments: tuple[str, ...],
        expected: int | None,
    ) -> None:
        converter = mocker.patch("spatialdata_io.converters.generic_to_zarr.generic_to_zarr")
        image = tmp_path / "image.tif"
        image.touch()

        result = runner.invoke(
            generic_command,
            ["--input", str(image), "--output", str(tmp_path / "output.zarr"), "--data-axes", "yx", *arguments],
        )

        assert result.exit_code == 0, result.output
        # Omitted chunks reach the reader, whose default depends on the canonical axes.
        assert converter.call_args.kwargs["chunks"] == expected
        assert converter.call_args.kwargs["scale_factors"] == DEFAULT_SCALE_FACTORS
