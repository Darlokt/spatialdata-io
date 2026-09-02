"""Test the Visium command boundary."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from spatialdata_io._cli.commands.visium import visium_command

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestVisiumCommand:
    """Verify exact typed option forwarding without decoder internals."""

    def test_forwards_supported_options_and_writes_output(
        self,
        tmp_path: Path,
        runner: CliRunner,
        mocker: MockerFixture,
        reader_result: MagicMock,
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.visium.visium", return_value=reader_result)
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
                "--counts-source",
                "raw_feature_bc_matrix.h5",
                "--fullres-image-file",
                "image.tif",
                "--positions-file",
                "positions.csv",
                "--scale-factors-file",
                "scales.json",
                "--no-make-var-names-unique",
                "--data-axes",
                "ZCYX",
                "--tiff-series",
                "2",
                "--chunks",
                "1",
                "--chunks",
                "256",
                "--chunks",
                "256",
            ],
        )

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            dataset_id="sample",
            counts_source=Path("raw_feature_bc_matrix.h5"),
            fullres_image_file=Path("image.tif"),
            positions_file=Path("positions.csv"),
            scale_factors_file=Path("scales.json"),
            make_var_names_unique=False,
            data_axes="ZCYX",
            tiff_series=2,
            chunks=(1, 256, 256),
        )
        reader_result.write.assert_called_once_with(output)

    def test_option_types_and_defaults_match_reader_contract(self) -> None:
        options = {parameter.name: parameter for parameter in visium_command.params}

        assert list(options) == [
            "input_path",
            "output_path",
            "dataset_id",
            "counts_source",
            "fullres_image_file",
            "positions_file",
            "scale_factors_file",
            "make_var_names_unique",
            "data_axes",
            "tiff_series",
            "chunks",
        ]
        for name in ("counts_source", "fullres_image_file", "positions_file", "scale_factors_file"):
            option_type = options[name].type
            assert isinstance(option_type, click.Path)
            assert option_type.type is Path
        assert options["make_var_names_unique"].default is True
        assert options["data_axes"].default is None
        assert options["tiff_series"].default is None
        assert options["chunks"].multiple
        assert options["chunks"].default is None

    def test_normalizes_one_chunk_value(self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture) -> None:
        reader = mocker.patch("spatialdata_io.readers.visium.visium")

        result = runner.invoke(
            visium_command,
            ["--input", str(tmp_path), "--output", str(tmp_path / "out"), "--chunks", "256"],
        )

        assert result.exit_code == 0, result.output
        assert reader.call_args.kwargs["chunks"] == 256

    def test_rejects_two_chunk_values(self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture) -> None:
        reader = mocker.patch("spatialdata_io.readers.visium.visium")

        result = runner.invoke(
            visium_command,
            [
                "--input",
                str(tmp_path),
                "--output",
                str(tmp_path / "out"),
                "--chunks",
                "1",
                "--chunks",
                "256",
            ],
        )

        assert result.exit_code == 2
        assert "passed once for all axes or three times" in result.output
        reader.assert_not_called()
