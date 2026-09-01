"""Test the DBiT command boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata_io._cli.commands.dbit import dbit_command

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestDbitCommand:
    """Verify artifact conversion and the typed reader call."""

    def test_forwards_options_and_writes_output(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.dbit.dbit", return_value=reader_result)
        table = tmp_path / "table.h5ad"
        barcodes = tmp_path / "barcodes.tsv"
        image = tmp_path / "image.tif"
        for artifact in (table, barcodes, image):
            artifact.touch()
        output = tmp_path / "output.zarr"

        result = runner.invoke(
            dbit_command,
            [
                "--input",
                str(tmp_path),
                "--output",
                str(output),
                "--anndata-path",
                str(table),
                "--barcode-position",
                str(barcodes),
                "--image-path",
                str(image),
                "--dataset-id",
                "sample",
                "--border",
                "false",
                "--border-scale",
                "1.5",
            ],
        )

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            anndata_path=str(table),
            barcode_position=str(barcodes),
            image_path=str(image),
            dataset_id="sample",
            border=False,
            border_scale=1.5,
        )
        reader_result.write.assert_called_once_with(output)
