"""Typed CosMx command options and exact reader forwarding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io._cli.commands.cosmx import cosmx_command
from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestCosmxCommand:
    """Keep prefix, transcript and raster options aligned with the reader."""

    def test_default_options(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.cosmx.cosmx", return_value=reader_result)
        output = tmp_path / "output.zarr"
        result = runner.invoke(cosmx_command, ["--input", str(tmp_path), "--output", str(output)])
        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            dataset_id=None,
            include_transcripts=True,
            scale_factors=DEFAULT_SCALE_FACTORS,
            chunks=DEFAULT_CHUNKS,
        )
        reader_result.write.assert_called_once_with(output)

    def test_explicit_options(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.cosmx.cosmx", return_value=reader_result)
        arguments = [
            "--dataset-id",
            "Lung5_Rep2",
            "--no-include-transcripts",
            "--scale-factors",
            "2",
            "--chunks",
            "3",
            "--chunks",
            "512",
            "--chunks",
            "512",
        ]
        result = runner.invoke(
            cosmx_command, ["--input", str(tmp_path), "--output", str(tmp_path / "out.zarr"), *arguments]
        )
        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path, dataset_id="Lung5_Rep2", include_transcripts=False, scale_factors=(2,), chunks=(3, 512, 512)
        )

    def test_scalar_chunks(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.cosmx.cosmx", return_value=reader_result)
        result = runner.invoke(
            cosmx_command, ["--input", str(tmp_path), "--output", str(tmp_path / "out.zarr"), "--chunks", "64"]
        )
        assert result.exit_code == 0, result.output
        assert reader.call_args.kwargs["chunks"] == 64

    @pytest.mark.parametrize(
        "arguments",
        [
            ["--chunks", "1", "--chunks", "2"],
            ["--chunks", "0"],
            ["--scale-factors", "1"],
            ["--transcripts", "true"],
            ["--imread-kwargs", "{}"],
            ["--image-models-kwargs", "{}"],
        ],
    )
    def test_invalid_or_removed_options_fail_before_reader(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, arguments: list[str]
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.cosmx.cosmx")
        result = runner.invoke(
            cosmx_command, ["--input", str(tmp_path), "--output", str(tmp_path / "out.zarr"), *arguments]
        )
        assert result.exit_code == 2, result.output
        reader.assert_not_called()

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cosmx_command, ["--help"])
        assert result.exit_code == 0
        for option in ("--dataset-id", "--include-transcripts", "--scale-factors", "--chunks"):
            assert option in result.output
