"""Typed MCMICRO command options and exact reader forwarding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io._cli.commands.mcmicro import mcmicro_command

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestMcmicroCommand:
    """Keep raster options aligned with the reader and reject removed mappings."""

    def test_default_options(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.mcmicro.mcmicro", return_value=reader_result)
        output = tmp_path / "output.zarr"
        result = runner.invoke(mcmicro_command, ["--input", str(tmp_path), "--output", str(output)])
        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(tmp_path, scale_factors=None, chunks=None)
        reader_result.write.assert_called_once_with(output)

    def test_explicit_options_keep_order(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.mcmicro.mcmicro", return_value=reader_result)
        arguments = [
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
        result = runner.invoke(
            mcmicro_command, ["--input", str(tmp_path), "--output", str(tmp_path / "output.zarr"), *arguments]
        )
        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(tmp_path, scale_factors=(2, 3), chunks=(1, 16, 32))

    def test_scalar_chunks(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.mcmicro.mcmicro", return_value=reader_result)
        result = runner.invoke(
            mcmicro_command, ["--input", str(tmp_path), "--output", str(tmp_path / "output.zarr"), "--chunks", "64"]
        )
        assert result.exit_code == 0, result.output
        assert reader.call_args.kwargs["chunks"] == 64

    @pytest.mark.parametrize(
        "arguments",
        [
            ["--chunks", "1", "--chunks", "2"],
            ["--chunks", "0"],
            ["--scale-factors", "1"],
            ["--imread-kwargs", "{}"],
            ["--image-models-kwargs", "{}"],
            ["--labels-models-kwargs", "{}"],
        ],
    )
    def test_invalid_or_removed_options_fail_before_reader(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, arguments: list[str]
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.mcmicro.mcmicro")
        result = runner.invoke(
            mcmicro_command, ["--input", str(tmp_path), "--output", str(tmp_path / "output.zarr"), *arguments]
        )
        assert result.exit_code == 2, result.output
        reader.assert_not_called()

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(mcmicro_command, ["--help"])
        assert result.exit_code == 0
        for option in ("--scale-factors", "--chunks"):
            assert option in result.output
