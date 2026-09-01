"""Test the MACSima command boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata_io._cli.commands.macsima import macsima_command

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestMacsimaCommand:
    """Verify repeated options and the typed reader call."""

    def test_forwards_options_and_writes_output(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.macsima.macsima", return_value=reader_result)
        output = tmp_path / "output.zarr"

        result = runner.invoke(
            macsima_command,
            [
                "--input",
                str(tmp_path),
                "--output",
                str(output),
                "--parsing-style",
                "processed_multiple_folders",
                "--filter-folder-names",
                "skip-a",
                "--filter-folder-names",
                "skip-b",
                "--subset",
                "64",
                "--c-subset",
                "3",
                "--max-chunk-size",
                "128",
                "--c-chunks-size",
                "2",
                "--multiscale",
                "false",
                "--transformations",
                "false",
                "--scale-factors",
                "2",
                "--scale-factors",
                "4",
                "--default-scale-factor",
                "3",
                "--nuclei-channel-name",
                "Hoechst",
                "--split-threshold-nuclei-channel",
                "4",
                "--skip-rounds",
                "1",
                "--skip-rounds",
                "3",
                "--include-cycle-in-channel-name",
                "true",
                "--imread-kwargs",
                '{"n": 2}',
            ],
        )

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            path=tmp_path,
            parsing_style="processed_multiple_folders",
            filter_folder_names=["skip-a", "skip-b"],
            imread_kwargs={"n": 2},
            subset=64,
            c_subset=3,
            max_chunk_size=128,
            c_chunks_size=2,
            multiscale=False,
            transformations=False,
            scale_factors=[2, 4],
            default_scale_factor=3,
            nuclei_channel_name="Hoechst",
            split_threshold_nuclei_channel=4,
            skip_rounds=[1, 3],
            include_cycle_in_channel_name=True,
        )
        reader_result.write.assert_called_once_with(output)
