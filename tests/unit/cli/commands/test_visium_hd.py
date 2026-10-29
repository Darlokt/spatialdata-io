"""Test Visium HD CLI conversion, defaults and explicit product selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io._cli.commands.visium_hd import visium_hd_command
from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestVisiumHdCommand:
    """Verify typed repeated, tri-state and paired flag options."""

    def test_defaults(self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock) -> None:
        reader = mocker.patch("spatialdata_io.readers.visium_hd.visium_hd", return_value=reader_result)
        output = tmp_path / "output.zarr"
        result = runner.invoke(visium_hd_command, ["--input", str(tmp_path), "--output", str(output)])
        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            bin_sizes="all",
            bin_geometry="squares",
            bin_counts_kind="filtered",
            include_cell_boundaries=None,
            include_nucleus_boundaries=None,
            include_nucleus_table=False,
            cell_counts_kind="filtered",
            invalid_geometries="keep",
            include_images=True,
            include_cytassist_image=False,
            fullres_image_file=None,
            data_axes=None,
            tiff_series=None,
            make_var_names_unique=True,
            scale_factors=DEFAULT_SCALE_FACTORS,
            chunks=DEFAULT_CHUNKS,
        )
        reader_result.write.assert_called_once_with(output)

    @pytest.mark.parametrize("selection", [("8", "2"), ("none",)])
    def test_selection(
        self,
        tmp_path: Path,
        runner: CliRunner,
        mocker: MockerFixture,
        reader_result: MagicMock,
        selection: tuple[str, ...],
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.visium_hd.visium_hd", return_value=reader_result)
        args = [
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.zarr"),
            "--bin-geometry",
            "labels",
            "--no-include-cell-boundaries",
            "--include-nucleus-boundaries",
            "--include-nucleus-table",
            "--cell-counts-kind",
            "raw",
            "--bin-counts-kind",
            "raw",
            "--no-include-images",
            "--data-axes",
            "c",
            "--data-axes",
            "y",
            "--data-axes",
            "x",
            "--tiff-series",
            "1",
        ]
        for size in selection:
            args.extend(["--bin-sizes", size])
        result = runner.invoke(visium_hd_command, args)
        assert result.exit_code == 0, result.output
        assert reader.call_args.kwargs["bin_sizes"] == ((2, 8) if selection[0] == "8" else ())
        assert reader.call_args.kwargs["data_axes"] == ("c", "y", "x")
        assert reader.call_args.kwargs["include_cell_boundaries"] is False
        assert reader.call_args.kwargs["include_nucleus_table"] is True

    @pytest.mark.parametrize("sizes", [("all", "8"), ("3",), ("0",), ("8", "8"), ("bad",)])
    def test_rejects_bad_sizes(self, tmp_path: Path, runner: CliRunner, sizes: tuple[str, ...]) -> None:
        args = ["--input", str(tmp_path), "--output", str(tmp_path / "out.zarr")]
        for size in sizes:
            args.extend(["--bin-sizes", size])
        result = runner.invoke(visium_hd_command, args)
        assert result.exit_code == 2
