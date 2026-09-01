"""Test the seqFISH command boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata_io._cli.commands.seqfish import seqfish_command

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestSeqfishCommand:
    """Verify repeated and boolean option conversion."""

    def test_forwards_options_and_writes_output(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.seqfish.seqfish", return_value=reader_result)
        output = tmp_path / "output.zarr"

        result = runner.invoke(
            seqfish_command,
            [
                "--input",
                str(tmp_path),
                "--output",
                str(output),
                "--load-images",
                "false",
                "--load-labels",
                "false",
                "--load-points",
                "false",
                "--load-shapes",
                "false",
                "--cells-as-circles",
                "true",
                "--rois",
                "roi-a",
                "--rois",
                "roi-b",
                "--raster-models-scale-factors",
                "2",
                "--raster-models-scale-factors",
                "4",
                "--imread-kwargs",
                '{"n": 2}',
            ],
        )

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            load_images=False,
            load_labels=False,
            load_points=False,
            load_shapes=False,
            cells_as_circles=True,
            rois=["roi-a", "roi-b"],
            raster_models_scale_factors=[2, 4],
            imread_kwargs={"n": 2},
        )
        reader_result.write.assert_called_once_with(output)
