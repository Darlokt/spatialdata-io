"""Test the Curio command boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io._cli.commands.curio import curio_command

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture

pytestmark = pytest.mark.curio


class TestCurioCommand:
    """Verify typed paths and output writing."""

    def test_calls_reader_and_writes_output(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.curio.curio", return_value=reader_result)
        output = tmp_path / "output.zarr"

        result = runner.invoke(curio_command, ["--input", str(tmp_path), "--output", str(output)])

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(tmp_path, path_policy="strict")
        reader_result.write.assert_called_once_with(output)
