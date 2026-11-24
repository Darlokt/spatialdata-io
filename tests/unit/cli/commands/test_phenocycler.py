"""Check region, channel and bit-depth selection in the PhenoCycler command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io._cli.commands.phenocycler import phenocycler_command
from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


@pytest.mark.phenocycler
class TestPhenoCyclerCommand:
    def test_file_and_selection(self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture) -> None:
        source = tmp_path / "sample.qptiff"
        source.touch()
        output = tmp_path / "out.zarr"
        reader = mocker.patch("spatialdata_io.readers.phenocycler.phenocycler")

        result = runner.invoke(
            phenocycler_command,
            [
                "-i",
                str(source),
                "-o",
                str(output),
                "--regions",
                "sample",
                "--channel-names",
                "DNA",
                "--channel-names",
                "CD3",
                "--read-all-bit-depths",
                "--chunks",
                "64",
            ],
        )
        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            source,
            regions=("sample",),
            channel_names=("DNA", "CD3"),
            read_all_bit_depths=True,
            chunks=64,
            scale_factors=(),
            labels=None,
            labels_to_image=None,
            path_policy="strict",
        )
        reader.return_value.write.assert_called_once_with(output)

    def test_directory_defaults(self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture) -> None:
        reader = mocker.patch("spatialdata_io.readers.phenocycler.phenocycler")
        result = runner.invoke(phenocycler_command, ["-i", str(tmp_path), "-o", str(tmp_path / "out")])
        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            regions=None,
            channel_names=None,
            read_all_bit_depths=False,
            chunks=DEFAULT_CHUNKS,
            scale_factors=DEFAULT_SCALE_FACTORS,
            labels=None,
            labels_to_image=None,
            path_policy="strict",
        )
