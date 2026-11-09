"""Test the Generic command boundary."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import click
import pytest

from spatialdata_io._cli.commands.generic import generic_command
from spatialdata_io._raster_defaults import DEFAULT_SCALE_FACTORS
from spatialdata_io.converters.generic_to_zarr import generic_to_zarr
from spatialdata_io.readers.generic import generic, image

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


@pytest.mark.generic
class TestGenericCommand:
    """Verify Click conversion to the Generic converter contract."""

    def test_option_order_follows_the_converter_signature(self) -> None:
        options = [parameter.name for parameter in generic_command.params]
        _path, _output, _name, *reader_options = inspect.signature(generic_to_zarr).parameters

        assert options == ["input_path", "output_path", "element_name", *reader_options]

    def test_data_axes_is_one_string_like_the_readers(self) -> None:
        option = next(parameter for parameter in generic_command.params if parameter.name == "data_axes")

        assert option.type is click.STRING
        assert not option.multiple
        assert inspect.signature(generic).parameters["data_axes"].annotation == "str | None"
        assert inspect.signature(image).parameters["data_axes"].annotation == "str"
        assert inspect.signature(generic_to_zarr).parameters["data_axes"].annotation == "str | None"

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
        assert converter.call_args.kwargs["tiff_series"] is None
