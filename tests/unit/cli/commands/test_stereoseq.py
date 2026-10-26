"""Exact Stereo-seq CLI selections, cardinality and output ownership."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click
import pytest

from spatialdata_io._cli.commands.stereoseq import stereoseq_command

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestStereoseqCommand:
    """Keep explicit product selection aligned with the reader signature."""

    def test_forwards_options_and_writes_output(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.stereoseq.stereoseq", return_value=reader_result)
        output = tmp_path / "output.zarr"
        result = runner.invoke(
            stereoseq_command,
            [
                "-i",
                str(tmp_path),
                "-o",
                str(output),
                "--sample-id",
                "sample.1",
                "--bin-sizes",
                "50",
                "--bin-sizes",
                "1",
                "--cell-bins",
                "both",
                "--cell-geometry",
                "polygons",
                "--square-counts-file",
                "sample.1.tissue.gef",
                "--cell-counts-file",
                "sample.1.cellbin.gef",
                "--adjusted-cell-counts-file",
                "sample.1.adjusted.cellbin.gef",
                "--analysis-files",
                "sample.1.cellbin_0.5.h5ad",
                "--analysis-files",
                "sample.1.cellbin_1.h5ad",
                "--image-roles",
                "registered",
                "--image-roles",
                "mask",
                "--chunks",
                "1",
                "--chunks",
                "64",
                "--chunks",
                "32",
            ],
        )
        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            sample_id="sample.1",
            bin_sizes=(1, 50),
            cell_bins="both",
            cell_geometry="polygons",
            square_counts_file=Path("sample.1.tissue.gef"),
            cell_counts_file=Path("sample.1.cellbin.gef"),
            adjusted_cell_counts_file=Path("sample.1.adjusted.cellbin.gef"),
            analysis_files=(Path("sample.1.cellbin_0.5.h5ad"), Path("sample.1.cellbin_1.h5ad")),
            image_roles=("registered", "mask"),
            chunks=(1, 64, 32),
        )
        reader_result.write.assert_called_once_with(output)

    @pytest.mark.parametrize("sizes", [(), ("128",)])
    def test_all_and_default_options(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock, sizes: tuple[str, ...]
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.stereoseq.stereoseq", return_value=reader_result)
        args = ["-i", str(tmp_path), "-o", str(tmp_path / "result"), "--bin-sizes", "all"]
        for size in sizes:
            args.extend(["--chunks", size])
        result = runner.invoke(stereoseq_command, args)
        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            sample_id=None,
            bin_sizes="all",
            cell_bins="none",
            cell_geometry="centroids",
            square_counts_file=None,
            cell_counts_file=None,
            adjusted_cell_counts_file=None,
            analysis_files=(),
            image_roles=(),
            chunks=128 if sizes else None,
        )

    @pytest.mark.parametrize(
        "options",
        [
            ["--bin-sizes", "all", "--bin-sizes", "1"],
            ["--bin-sizes", "0"],
            ["--bin-sizes", "1.5"],
            ["--bin-sizes", "1", "--bin-sizes", "01"],
            ["--bin-sizes", "\uff11"],
            ["--cell-bins", "cells"],
            ["--cell-geometry", "circles"],
            ["--image-roles", "labels"],
            ["--chunks", "0"],
            ["--chunks", "4", "--chunks", "8"],
            ["--imread-kwargs", "{}"],
            ["--dataset-id", "sample"],
        ],
    )
    def test_invalid_options_do_not_call_reader(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, options: list[str]
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.stereoseq.stereoseq")
        result = runner.invoke(stereoseq_command, ["-i", str(tmp_path), "-o", str(tmp_path / "result"), *options])
        assert result.exit_code != 0
        reader.assert_not_called()

    def test_option_defaults_and_path_kinds(self) -> None:
        options = {parameter.name: parameter for parameter in stereoseq_command.params}
        for name in ("square_counts_file", "cell_counts_file", "adjusted_cell_counts_file", "analysis_files"):
            option_type = options[name].type
            assert isinstance(option_type, click.Path)
            assert option_type.type is Path
        for name in ("bin_sizes", "image_roles", "analysis_files", "chunks"):
            assert options[name].multiple
            assert options[name].default == ()
        assert options["cell_bins"].default == "none"
        assert options["cell_geometry"].default == "centroids"
