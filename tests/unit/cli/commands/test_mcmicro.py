"""Test the MCMICRO command boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata_io._cli.commands.mcmicro import mcmicro_command

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestMcmicroCommand:
    """Verify legacy JSON conversion and output writing."""

    def test_forwards_options_and_writes_output(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.mcmicro.mcmicro", return_value=reader_result)
        output = tmp_path / "output.zarr"

        result = runner.invoke(
            mcmicro_command,
            [
                "--input",
                str(tmp_path),
                "--output",
                str(output),
                "--imread-kwargs",
                '{"n": 2}',
                "--image-models-kwargs",
                '{"scale_factors": null}',
                "--labels-models-kwargs",
                '{"scale_factors": null}',
            ],
        )

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            imread_kwargs={"n": 2},
            image_models_kwargs={"scale_factors": None},
            labels_models_kwargs={"scale_factors": None},
        )
        reader_result.write.assert_called_once_with(output)
