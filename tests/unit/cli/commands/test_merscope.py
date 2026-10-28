"""Typed MERSCOPE command options and exact reader forwarding."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from spatialdata_io._cli.commands.merscope import merscope_command
from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestMerscopeCommand:
    """Keep inclusion, layer, chunk and source selection aligned with the reader."""

    def test_default_options(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.merscope.merscope", return_value=reader_result)
        output = tmp_path / "output.zarr"
        result = runner.invoke(merscope_command, ["--input", str(tmp_path), "--output", str(output)])
        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            vpt_directory=None,
            z_layers=None,
            region_name=None,
            slide_name=None,
            include_transcripts=True,
            include_cell_boundaries=True,
            include_table=True,
            include_images=True,
            scale_factors=DEFAULT_SCALE_FACTORS,
            chunks=DEFAULT_CHUNKS,
        )
        reader_result.write.assert_called_once_with(output)

    def test_explicit_options_keep_order_and_relative_vpt_path(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.merscope.merscope", return_value=reader_result)
        output = tmp_path / "output.zarr"
        arguments = [
            "--input",
            str(tmp_path),
            "--output",
            str(output),
            "--vpt-directory",
            "analysis",
            "--z-layers",
            "2",
            "--z-layers",
            "0",
            "--region-name",
            "region",
            "--slide-name",
            "slide",
            "--no-include-transcripts",
            "--no-include-cell-boundaries",
            "--no-include-table",
            "--no-include-images",
            "--scale-factors",
            "2",
            "--scale-factors",
            "3",
            "--chunks",
            "1",
            "--chunks",
            "16",
            "--chunks",
            "32",
        ]
        result = runner.invoke(merscope_command, arguments)
        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            vpt_directory=Path("analysis"),
            z_layers=(2, 0),
            region_name="region",
            slide_name="slide",
            include_transcripts=False,
            include_cell_boundaries=False,
            include_table=False,
            include_images=False,
            scale_factors=(2, 3),
            chunks=(1, 16, 32),
        )
        reader_result.write.assert_called_once_with(output)

    def test_scalar_chunks(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.merscope.merscope", return_value=reader_result)
        result = runner.invoke(
            merscope_command, ["--input", str(tmp_path), "--output", str(tmp_path / "output.zarr"), "--chunks", "64"]
        )
        assert result.exit_code == 0, result.output
        assert reader.call_args.kwargs["chunks"] == 64

    @pytest.mark.parametrize(
        "arguments",
        [
            ["--chunks", "1", "--chunks", "2"],
            ["--chunks", "0"],
            ["--z-layers", "-1"],
            ["--scale-factors", "1"],
            ["--no-include-table", "false"],
            ["--backend", "rioxarray"],
            ["--imread-kwargs", "{}"],
            ["--image-models-kwargs", "{}"],
            ["--vpt-outputs", "analysis"],
            ["--transcripts", "true"],
            ["--cells-boundaries", "true"],
            ["--cells-table", "true"],
            ["--mosaic-images", "true"],
        ],
    )
    def test_invalid_or_removed_options_fail_before_reader(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, arguments: list[str]
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.merscope.merscope")
        result = runner.invoke(
            merscope_command, ["--input", str(tmp_path), "--output", str(tmp_path / "output.zarr"), *arguments]
        )
        assert result.exit_code == 2, result.output
        reader.assert_not_called()

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(merscope_command, ["--help"])
        assert result.exit_code == 0
        for option in ("--vpt-directory", "--z-layers", "--include-cell-boundaries", "--scale-factors", "--chunks"):
            assert option in result.output
