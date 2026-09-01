"""Test the experimental ISS command boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata_io._cli.commands.iss import iss_command

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestIssCommand:
    """Verify artifact paths and legacy option conversion."""

    def test_forwards_options_and_writes_output(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.iss.iss", return_value=reader_result)
        raw = tmp_path / "raw.tif"
        labels = tmp_path / "labels.tif"
        table = tmp_path / "table.h5ad"
        for artifact in (raw, labels, table):
            artifact.touch()
        output = tmp_path / "output.zarr"

        result = runner.invoke(
            iss_command,
            [
                "--input",
                str(tmp_path),
                "--output",
                str(output),
                "--raw-relative-path",
                str(raw),
                "--labels-relative-path",
                str(labels),
                "--h5ad-relative-path",
                str(table),
                "--instance-key",
                "cell_id",
                "--dataset-id",
                "sample",
                "--multiscale-image",
                "false",
                "--multiscale-labels",
                "false",
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
            raw,
            labels,
            table,
            instance_key="cell_id",
            dataset_id="sample",
            multiscale_image=False,
            multiscale_labels=False,
            imread_kwargs={"n": 2},
            image_models_kwargs={"scale_factors": None},
            labels_models_kwargs={"scale_factors": None},
        )
        reader_result.write.assert_called_once_with(output)
