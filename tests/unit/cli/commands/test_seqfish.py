"""seqFISH paired inclusion flags, finite choices and repeated cardinality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io._cli.commands.seqfish import seqfish_command

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestSeqfishCommand:
    """Mirror reader defaults and normalize CLI values exactly once."""

    def test_defaults_forward_and_write(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.seqfish.seqfish", return_value=reader_result)
        output = tmp_path / "output.zarr"
        result = runner.invoke(seqfish_command, ["--input", str(tmp_path), "--output", str(output)])
        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            include_images=True,
            include_labels=True,
            include_transcripts=True,
            include_shapes=True,
            shape_geometry="polygons",
            rois=None,
            scale_factors=None,
            chunks=None,
        )
        reader_result.write.assert_called_once_with(output)

    @pytest.mark.parametrize("geometry", ["polygons", "circles", "both"])
    def test_paired_flags_repeated_order_and_geometry_choices(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock, geometry: str
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.seqfish.seqfish", return_value=reader_result)
        result = runner.invoke(
            seqfish_command,
            [
                "--input",
                str(tmp_path),
                "--output",
                str(tmp_path / "out.zarr"),
                "--no-include-images",
                "--no-include-labels",
                "--no-include-transcripts",
                "--no-include-shapes",
                "--shape-geometry",
                geometry,
                "--rois",
                "Roi2",
                "--rois",
                "Roi1",
                "--scale-factors",
                "2",
                "--scale-factors",
                "3",
                "--chunks",
                "1",
                "--chunks",
                "16",
                "--chunks",
                "32",
            ],
        )
        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            include_images=False,
            include_labels=False,
            include_transcripts=False,
            include_shapes=False,
            shape_geometry=geometry,
            rois=("Roi2", "Roi1"),
            scale_factors=(2, 3),
            chunks=(1, 16, 32),
        )

    def test_scalar_chunks(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.seqfish.seqfish", return_value=reader_result)
        result = runner.invoke(
            seqfish_command, ["--input", str(tmp_path), "--output", str(tmp_path / "out.zarr"), "--chunks", "16"]
        )
        assert result.exit_code == 0, result.output
        assert reader.call_args.kwargs["chunks"] == 16

    @pytest.mark.parametrize(
        "arguments",
        [
            ["--chunks", "1", "--chunks", "2"],
            ["--chunks", "0"],
            ["--chunks", "{invalid-json}"],
            ["--shape-geometry", "invalid"],
            ["--scale-factors", "1"],
            ["--no-include-images", "false"],
            ["--load-images", "false"],
            ["--load-labels", "false"],
            ["--load-points", "false"],
            ["--load-shapes", "false"],
            ["--cells-as-circles", "true"],
            ["--imread-kwargs", "{}"],
            ["--raster-models-scale-factors", "2"],
        ],
    )
    def test_invalid_cardinality_values_and_removed_options_do_not_call_reader(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, arguments: list[str]
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.seqfish.seqfish")
        result = runner.invoke(
            seqfish_command, ["--input", str(tmp_path), "--output", str(tmp_path / "out.zarr"), *arguments]
        )
        assert result.exit_code == 2, result.output
        reader.assert_not_called()

    def test_help_lists_new_contract(self, runner: CliRunner) -> None:
        result = runner.invoke(seqfish_command, ["--help"])
        assert result.exit_code == 0
        for option in (
            "--include-images",
            "--no-include-images",
            "--shape-geometry",
            "--rois",
            "--scale-factors",
            "--chunks",
        ):
            assert option in result.output
