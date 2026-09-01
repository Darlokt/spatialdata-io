"""Test the CODEX command boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata_io._cli.commands.codex import codex_command

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestCodexCommand:
    """Verify Click conversion and the typed reader call."""

    def test_forwards_options_and_writes_output(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.codex.codex", return_value=reader_result)
        output = tmp_path / "output.zarr"

        result = runner.invoke(
            codex_command,
            ["--input", str(tmp_path), "--output", str(output), "--fcs", "false", "--imread-kwargs", '{"n": 2}'],
        )

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(tmp_path, fcs=False, imread_kwargs={"n": 2})
        reader_result.write.assert_called_once_with(output)
