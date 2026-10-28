"""Test the MACSima command boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io._cli.commands.macsima import macsima_command
from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestMacsimaCommand:
    """Verify repeated options, typed values and the normalized reader call."""

    def test_forwards_options_and_writes_output(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.macsima.macsima", return_value=reader_result)
        output = tmp_path / "output.zarr"

        result = runner.invoke(
            macsima_command,
            [
                *("--input", str(tmp_path), "--output", str(output)),
                *("--parsing-style", "processed_multiple_folders"),
                *("--exclude-folder-names", "skip-a", "--exclude-folder-names", "skip b"),
                *("--exclude-cycles", "1", "--exclude-cycles", "3"),
                "--include-cycle-in-channel-names",
                *("--nuclei-channel-name", "Hoechst"),
                *("--split-threshold-nuclei-channel", "4"),
                "--no-include-transformations",
                *("--scale-factors", "2", "--scale-factors", "4"),
                *("--chunks", "1", "--chunks", "256", "--chunks", "512"),
            ],
        )

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            parsing_style="processed_multiple_folders",
            exclude_folder_names=("skip-a", "skip b"),
            exclude_cycles=(1, 3),
            include_cycle_in_channel_names=True,
            nuclei_channel_name="Hoechst",
            split_threshold_nuclei_channel=4,
            include_transformations=False,
            scale_factors=(2, 4),
            chunks=(1, 256, 512),
        )
        reader_result.write.assert_called_once_with(output)

    def test_defaults_match_the_reader(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.macsima.macsima", return_value=reader_result)

        result = runner.invoke(macsima_command, ["--input", str(tmp_path), "--output", str(tmp_path / "out.zarr")])

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            parsing_style="processed_single_folder",
            exclude_folder_names=(),
            exclude_cycles=(),
            include_cycle_in_channel_names=False,
            nuclei_channel_name="DAPI",
            split_threshold_nuclei_channel=2,
            include_transformations=True,
            scale_factors=DEFAULT_SCALE_FACTORS,
            chunks=DEFAULT_CHUNKS,
        )

    @pytest.mark.parametrize("case", [("none", None), ("NONE", None), ("0", 0), ("7", 7)])
    def test_split_threshold_accepts_counts_or_none(
        self,
        tmp_path: Path,
        runner: CliRunner,
        mocker: MockerFixture,
        reader_result: MagicMock,
        case: tuple[str, int | None],
    ) -> None:
        value, expected = case
        reader = mocker.patch("spatialdata_io.readers.macsima.macsima", return_value=reader_result)

        result = runner.invoke(
            macsima_command,
            [
                *("--input", str(tmp_path), "--output", str(tmp_path / "out.zarr")),
                *("--split-threshold-nuclei-channel", value),
            ],
        )

        assert result.exit_code == 0, result.output
        assert reader.call_args.kwargs["split_threshold_nuclei_channel"] == expected

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (("--split-threshold-nuclei-channel", "-1"), "neither a non-negative integer nor 'none'"),
            (("--split-threshold-nuclei-channel", "two"), "neither a non-negative integer nor 'none'"),
            (("--chunks", "1", "--chunks", "2"), "Pass --chunks once for every axis"),
            (("--exclude-cycles", "-1"), "is not in the range"),
            (("--scale-factors", "1"), "is not in the range"),
        ],
    )
    def test_rejects_invalid_values_before_reading(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, arguments: tuple[str, ...], message: str
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.macsima.macsima")

        result = runner.invoke(
            macsima_command, ["--input", str(tmp_path), "--output", str(tmp_path / "out.zarr"), *arguments]
        )

        assert result.exit_code == 2
        assert message in result.output
        reader.assert_not_called()

    def test_single_chunk_size_applies_to_every_axis(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.macsima.macsima", return_value=reader_result)

        result = runner.invoke(
            macsima_command, ["--input", str(tmp_path), "--output", str(tmp_path / "out.zarr"), "--chunks", "64"]
        )

        assert result.exit_code == 0, result.output
        assert reader.call_args.kwargs["chunks"] == 64
