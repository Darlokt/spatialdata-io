"""Test the normalized ISS command boundary."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from spatialdata_io._cli.commands.iss import iss_command

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestIssCommand:
    """Verify paths, optional sequences, cardinality and output writing."""

    def test_defaults_resolve_artifacts_in_reader(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.iss.iss", return_value=reader_result)
        output = tmp_path / "out.zarr"
        result = runner.invoke(
            iss_command,
            [
                "-i",
                str(tmp_path),
                "-o",
                str(output),
                "--raw-image-file",
                "raw.tiff",
                "--labels-file",
                "labels.tiff",
                "--table-file",
                "cells.h5ad",
            ],
        )
        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            raw_image_file=Path("raw.tiff"),
            labels_file=Path("labels.tiff"),
            table_file=Path("cells.h5ad"),
            instance_key=None,
            dataset_id="region",
            image_scale_factors=None,
            labels_scale_factors=None,
            data_axes=None,
            chunks=None,
        )
        reader_result.write.assert_called_once_with(output)

    @pytest.mark.parametrize("chunk_args", [["12"], ["1", "8", "16"]])
    def test_explicit_options(
        self,
        tmp_path: Path,
        runner: CliRunner,
        mocker: MockerFixture,
        reader_result: MagicMock,
        chunk_args: list[str],
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.iss.iss", return_value=reader_result)
        expected = 12 if len(chunk_args) == 1 else (1, 8, 16)
        args = [
            "-i",
            str(tmp_path),
            "-o",
            str(tmp_path / "out.zarr"),
            "--raw-image-file",
            "raw.tiff",
            "--labels-file",
            "labels.tiff",
            "--table-file",
            "cells.h5ad",
            "--instance-key",
            "cell_id",
            "--dataset-id",
            "sample",
            "--data-axes",
            "CYX",
            "--image-scale-factors",
            "2",
            "--image-scale-factors",
            "3",
            "--labels-scale-factors",
            "2",
        ]
        for value in chunk_args:
            args.extend(["--chunks", value])
        result = runner.invoke(iss_command, args)
        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            raw_image_file=Path("raw.tiff"),
            labels_file=Path("labels.tiff"),
            table_file=Path("cells.h5ad"),
            instance_key="cell_id",
            dataset_id="sample",
            image_scale_factors=(2, 3),
            labels_scale_factors=(2,),
            data_axes="CYX",
            chunks=expected,
        )

    @pytest.mark.parametrize("omitted", ["--raw-image-file", "--labels-file", "--table-file"])
    def test_required_artifacts(self, tmp_path: Path, runner: CliRunner, omitted: str) -> None:
        args = ["-i", str(tmp_path), "-o", str(tmp_path / "out.zarr")]
        for option, value in (
            ("--raw-image-file", "raw.tiff"),
            ("--labels-file", "labels.tiff"),
            ("--table-file", "cells.h5ad"),
        ):
            if option != omitted:
                args.extend([option, value])
        result = runner.invoke(iss_command, args)
        assert result.exit_code == 2
        assert omitted in result.output

    @pytest.mark.parametrize(
        "extra",
        [
            ["--chunks", "1", "--chunks", "2"],
            ["--chunks", "0"],
            ["--labels-scale-factors", "1"],
            ["--image-scale-factors", "1"],
            ["--imread-kwargs", "{}"],
        ],
    )
    def test_invalid_options(self, tmp_path: Path, runner: CliRunner, extra: list[str]) -> None:
        result = runner.invoke(
            iss_command,
            [
                "-i",
                str(tmp_path),
                "-o",
                str(tmp_path / "out.zarr"),
                "--raw-image-file",
                "raw.tiff",
                "--labels-file",
                "labels.tiff",
                "--table-file",
                "cells.h5ad",
                *extra,
            ],
        )
        assert result.exit_code == 2
