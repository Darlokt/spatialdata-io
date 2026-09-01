"""Test the MERSCOPE command boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata_io._cli.commands.merscope import merscope_command

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestMerscopeCommand:
    """Verify path, boolean, choice, and JSON conversion."""

    def test_forwards_options_and_writes_output(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.merscope.merscope", return_value=reader_result)
        vpt = tmp_path / "vpt"
        vpt.mkdir()
        output = tmp_path / "output.zarr"

        result = runner.invoke(
            merscope_command,
            [
                "--input",
                str(tmp_path),
                "--output",
                str(output),
                "--vpt-outputs",
                str(vpt),
                "--z-layers",
                "1",
                "--region-name",
                "region",
                "--slide-name",
                "slide",
                "--backend",
                "rioxarray",
                "--transcripts",
                "false",
                "--cells-boundaries",
                "false",
                "--cells-table",
                "false",
                "--mosaic-images",
                "false",
                "--imread-kwargs",
                '{"n": 2}',
                "--image-models-kwargs",
                '{"scale_factors": null}',
            ],
        )

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            vpt_outputs=vpt,
            z_layers=1,
            region_name="region",
            slide_name="slide",
            backend="rioxarray",
            transcripts=False,
            cells_boundaries=False,
            cells_table=False,
            mosaic_images=False,
            imread_kwargs={"n": 2},
            image_models_kwargs={"scale_factors": None},
        )
        reader_result.write.assert_called_once_with(output)
