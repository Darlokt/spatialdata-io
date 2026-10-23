"""Verify the explicit CODEX Click surface and reader forwarding."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click
import pytest

from spatialdata_io._cli.commands.codex import codex_command

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestCodexCommand:
    """Keep option names, types, defaults, and chunk cardinality aligned."""

    def test_forwards_options_and_writes_output(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.codex.codex", return_value=reader_result)
        output = tmp_path / "output.zarr"

        result = runner.invoke(
            codex_command,
            [
                "--input",
                str(tmp_path),
                "--output",
                str(output),
                "--cell-table-file",
                "cells.fcs",
                "--image-file",
                "stitched.tif",
                "--data-axes",
                "CYX",
                "--tiff-series",
                "1",
                "--chunks",
                "1",
                "--chunks",
                "16",
                "--chunks",
                "32",
            ],
        )

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            cell_table_file=Path("cells.fcs"),
            image_file=Path("stitched.tif"),
            data_axes="CYX",
            tiff_series=1,
            chunks=(1, 16, 32),
        )
        reader_result.write.assert_called_once_with(output)

    @pytest.mark.parametrize(("values", "expected"), [([], None), (["--chunks", "16"], 16)])
    def test_chunk_default_and_scalar(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, values: list[str], expected: int | None
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.codex.codex")

        result = runner.invoke(codex_command, ["--input", str(tmp_path), "--output", str(tmp_path / "out"), *values])

        assert result.exit_code == 0, result.output
        assert reader.call_args.kwargs["chunks"] == expected

    @pytest.mark.parametrize(
        "values",
        [
            ["--chunks", "1", "--chunks", "2"],
            ["--chunks", "0"],
            ["--tiff-series", "-1"],
            ["--fcs", "false"],
            ["--imread-kwargs", "{}"],
        ],
    )
    def test_invalid_or_removed_options(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, values: list[str]
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.codex.codex")

        result = runner.invoke(codex_command, ["--input", str(tmp_path), "--output", str(tmp_path / "out"), *values])

        assert result.exit_code == 2
        reader.assert_not_called()

    def test_option_types_and_defaults(self) -> None:
        options = {parameter.name: parameter for parameter in codex_command.params}

        assert list(options) == [
            "input_path",
            "output_path",
            "cell_table_file",
            "image_file",
            "data_axes",
            "tiff_series",
            "chunks",
        ]
        for name in ("cell_table_file", "image_file"):
            option_type = options[name].type
            assert isinstance(option_type, click.Path)
            assert option_type.type is Path
            assert options[name].default is None
        assert options["data_axes"].default is None
        assert options["tiff_series"].default is None
        assert options["chunks"].multiple
        assert options["chunks"].default == ()
