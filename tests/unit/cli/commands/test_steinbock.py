"""Steinbock source choice, typed raster options, and persistence boundary."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import click
import pytest

from spatialdata_io._cli.commands.steinbock import steinbock_command
from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers.steinbock import steinbock

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestSteinbockCommand:
    """Align explicit defaults, source selection, sequence values, and chunk cardinality."""

    def test_defaults_and_output_writing(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.steinbock.steinbock", return_value=reader_result)
        output = tmp_path / "output.zarr"

        result = runner.invoke(steinbock_command, ["--input", str(tmp_path), "--output", str(output)])

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            labels_kind="deepcell",
            scale_factors=DEFAULT_SCALE_FACTORS,
            chunks=DEFAULT_CHUNKS,
        )
        reader_result.write.assert_called_once_with(output)

    @pytest.mark.parametrize("chunk_values", [["12"], ["1", "8", "16"]])
    def test_forwards_explicit_source_and_raster_options(
        self,
        tmp_path: Path,
        runner: CliRunner,
        mocker: MockerFixture,
        reader_result: MagicMock,
        chunk_values: list[str],
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.steinbock.steinbock", return_value=reader_result)
        output = tmp_path / "output.zarr"
        expected_chunks = 12 if len(chunk_values) == 1 else (1, 8, 16)
        arguments = [
            "--input",
            str(tmp_path),
            "--output",
            str(output),
            "--labels-kind",
            "ilastik",
            "--scale-factors",
            "2",
            "--scale-factors",
            "3",
        ]
        for value in chunk_values:
            arguments.extend(["--chunks", value])

        result = runner.invoke(steinbock_command, arguments)

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            labels_kind="ilastik",
            scale_factors=(2, 3),
            chunks=expected_chunks,
        )
        reader_result.write.assert_called_once_with(output)

    @pytest.mark.parametrize(
        "extra",
        [
            ["--labels-kind", "unknown"],
            ["--scale-factors", "1"],
            ["--image-scale-factors", "2"],
            ["--chunks", "0"],
            ["--chunks", "1", "--chunks", "2"],
            ["--chunks", "1", "--chunks", "2", "--chunks", "3", "--chunks", "4"],
            ["--scale-factors", "[2, 2]"],
            ["--imread-kwargs", "{}"],
            ["--image-models-kwargs", "{}"],
        ],
    )
    def test_invalid_and_removed_options(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, extra: list[str]
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.steinbock.steinbock")

        result = runner.invoke(
            steinbock_command, ["--input", str(tmp_path), "--output", str(tmp_path / "out.zarr"), *extra]
        )

        assert result.exit_code == 2
        reader.assert_not_called()

    def test_parameter_names_types_and_defaults(self) -> None:
        options = {parameter.name: parameter for parameter in steinbock_command.params}
        assert list(options) == [
            "input_path",
            "output_path",
            "labels_kind",
            "scale_factors",
            "chunks",
        ]
        reader_parameters = inspect.signature(steinbock).parameters
        assert set(reader_parameters) - {"path"} == set(options) - {"input_path", "output_path"}
        assert options["labels_kind"].default == reader_parameters["labels_kind"].default == "deepcell"
        choice = options["labels_kind"].type
        assert isinstance(choice, click.Choice)
        assert choice.choices == ("deepcell", "ilastik")
        assert options["scale_factors"].default == reader_parameters["scale_factors"].default == DEFAULT_SCALE_FACTORS
        assert options["chunks"].default == reader_parameters["chunks"].default == DEFAULT_CHUNKS
        for name in ("scale_factors", "chunks"):
            option = options[name]
            assert option.multiple
            option_type = option.type
            assert isinstance(option_type, click.IntRange)
            assert option_type.min == (1 if name == "chunks" else 2)
