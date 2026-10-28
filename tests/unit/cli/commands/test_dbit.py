"""Test DBiT source overrides, raster options, and public command alignment."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import click
import pytest

from spatialdata_io._cli.commands.dbit import dbit_command
from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers.dbit import dbit

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestDbitCommand:
    """Verify typed options, root-relative overrides, validation, and persistence."""

    def test_defaults_and_output_writing(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.dbit.dbit", return_value=reader_result)
        output = tmp_path / "output.zarr"

        result = runner.invoke(dbit_command, ["--input", str(tmp_path), "--output", str(output)])

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            table_file=None,
            barcode_positions_file=None,
            image_file=None,
            dataset_id=None,
            include_border=True,
            border_scale=1.0,
            data_axes=None,
            tiff_series=0,
            tiff_level=0,
            scale_factors=DEFAULT_SCALE_FACTORS,
            chunks=DEFAULT_CHUNKS,
        )
        reader_result.write.assert_called_once_with(output)

    @pytest.mark.parametrize("chunk_values", [["12"], ["1", "8", "16"]])
    @pytest.mark.parametrize("relative", [True, False])
    def test_forwards_source_and_raster_options(
        self,
        tmp_path: Path,
        runner: CliRunner,
        mocker: MockerFixture,
        chunk_values: list[str],
        *,
        relative: bool,
    ) -> None:
        reader_result = mocker.MagicMock()
        reader = mocker.patch("spatialdata_io.readers.dbit.dbit", return_value=reader_result)
        table = Path("expression.h5ad")
        barcodes = Path("positions.tsv")
        image = Path("histology.tif")
        if not relative:
            table, barcodes, image = (tmp_path / artifact for artifact in (table, barcodes, image))

        output = tmp_path / "output.zarr"
        arguments = [
            "--input",
            str(tmp_path),
            "--output",
            str(output),
            "--table-file",
            str(table),
            "--barcode-positions-file",
            str(barcodes),
            "--image-file",
            str(image),
            "--dataset-id",
            "sample",
            "--no-include-border",
            "--border-scale",
            "1.5",
            "--data-axes",
            "YXC",
            "--tiff-series",
            "2",
            "--tiff-level",
            "1",
            "--scale-factors",
            "2",
            "--scale-factors",
            "3",
        ]
        for value in chunk_values:
            arguments.extend(["--chunks", value])

        result = runner.invoke(dbit_command, arguments)

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            table_file=table,
            barcode_positions_file=barcodes,
            image_file=image,
            dataset_id="sample",
            include_border=False,
            border_scale=1.5,
            data_axes="YXC",
            tiff_series=2,
            tiff_level=1,
            scale_factors=(2, 3),
            chunks=12 if len(chunk_values) == 1 else (1, 8, 16),
        )
        reader_result.write.assert_called_once_with(output)

    def test_relative_overrides_are_not_validated_against_cwd(
        self,
        tmp_path: Path,
        runner: CliRunner,
        mocker: MockerFixture,
        reader_result: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.dbit.dbit", return_value=reader_result)
        root = tmp_path / "dataset"
        root.mkdir()

        # Same-named directories in cwd must not affect dataset-relative files.
        for name in ("table.h5ad", "positions.tsv", "image.png"):
            (root / name).touch()
            (tmp_path / name).mkdir()
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            dbit_command,
            [
                "--input",
                str(root),
                "--output",
                str(tmp_path / "out.zarr"),
                "--table-file",
                "table.h5ad",
                "--barcode-positions-file",
                "positions.tsv",
                "--image-file",
                "image.png",
            ],
        )

        assert result.exit_code == 0, result.output
        assert reader.call_args.kwargs["table_file"] == Path("table.h5ad")
        assert reader.call_args.kwargs["barcode_positions_file"] == Path("positions.tsv")
        assert reader.call_args.kwargs["image_file"] == Path("image.png")

    @pytest.mark.parametrize("scale", ["0", "3.99"])
    def test_border_scale_bounds_are_accepted(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock, scale: str
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.dbit.dbit", return_value=reader_result)

        result = runner.invoke(
            dbit_command,
            ["--input", str(tmp_path), "--output", str(tmp_path / "out.zarr"), "--border-scale", scale],
        )

        assert result.exit_code == 0, result.output
        assert reader.call_args.kwargs["border_scale"] == float(scale)

    @pytest.mark.parametrize("border_flag", ["--include-border", "--no-include-border"])
    @pytest.mark.parametrize("scale", ["-1", "4", "nan", "inf", "-inf"])
    def test_invalid_border_scale_is_rejected_even_when_border_disabled(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, border_flag: str, scale: str
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.dbit.dbit")

        result = runner.invoke(
            dbit_command,
            [
                "--input",
                str(tmp_path),
                "--output",
                str(tmp_path / "out.zarr"),
                border_flag,
                "--border-scale",
                scale,
            ],
        )

        assert result.exit_code == 2
        assert "--border-scale" in result.output
        reader.assert_not_called()

    @pytest.mark.parametrize(
        "extra",
        [
            ["--tiff-series", "-1"],
            ["--tiff-level", "-1"],
            ["--chunks", "1", "--chunks", "2"],
            ["--anndata-path", "table.h5ad"],
            ["--barcode-position", "positions.tsv"],
            ["--image-path", "image.png"],
            ["--border", "false"],
            ["--imread-kwargs", "{}"],
        ],
    )
    def test_invalid_and_removed_options(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, extra: list[str]
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.dbit.dbit")

        result = runner.invoke(dbit_command, ["--input", str(tmp_path), "--output", str(tmp_path / "out.zarr"), *extra])

        assert result.exit_code == 2
        reader.assert_not_called()

    def test_parameter_names_types_and_defaults(self) -> None:
        options = {parameter.name: parameter for parameter in dbit_command.params}
        reader_parameters = inspect.signature(dbit).parameters
        assert set(reader_parameters) - {"path"} == set(options) - {"input_path", "output_path"}

        for name in (
            "table_file",
            "barcode_positions_file",
            "image_file",
            "dataset_id",
            "include_border",
            "border_scale",
            "data_axes",
            "tiff_series",
            "tiff_level",
            "scale_factors",
            "chunks",
        ):
            assert options[name].default == reader_parameters[name].default

        for name in ("table_file", "barcode_positions_file", "image_file"):
            path_type = options[name].type
            assert isinstance(path_type, click.Path)
            assert path_type.type is Path
            assert not path_type.exists
            assert not path_type.readable

        border = options["include_border"]
        assert isinstance(border, click.Option)
        assert border.is_flag
        assert border.opts == ["--include-border"]
        assert border.secondary_opts == ["--no-include-border"]

        scale_type = options["border_scale"].type
        assert isinstance(scale_type, click.FloatRange)
        assert scale_type.min == 0
        assert scale_type.max == 4
        assert scale_type.max_open

        for name in ("tiff_series", "tiff_level"):
            option_type = options[name].type
            assert isinstance(option_type, click.IntRange)
            assert option_type.min == 0

        assert options["scale_factors"].default == DEFAULT_SCALE_FACTORS
        assert options["chunks"].default == DEFAULT_CHUNKS
