"""Test the Visium command boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata_io._cli.commands.visium import visium_command

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestVisiumCommand:
    """Verify artifact and legacy option conversion."""

    def test_forwards_options_and_writes_output(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.visium.visium", return_value=reader_result)
        image = tmp_path / "image.tif"
        positions = tmp_path / "positions.csv"
        scale_factors = tmp_path / "scales.json"
        for artifact in (image, positions, scale_factors):
            artifact.touch()
        output = tmp_path / "output.zarr"

        result = runner.invoke(
            visium_command,
            [
                "--input",
                str(tmp_path),
                "--output",
                str(output),
                "--dataset-id",
                "sample",
                "--counts-file",
                "raw_feature_bc_matrix.h5",
                "--fullres-image-file",
                str(image),
                "--tissue-positions-file",
                str(positions),
                "--scalefactors-file",
                str(scale_factors),
                "--var-names-make-unique",
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
            dataset_id="sample",
            counts_file="raw_feature_bc_matrix.h5",
            fullres_image_file=image,
            tissue_positions_file=positions,
            scalefactors_file=scale_factors,
            var_names_make_unique=False,
            imread_kwargs={"n": 2},
            image_models_kwargs={"scale_factors": None},
        )
        reader_result.write.assert_called_once_with(output)
