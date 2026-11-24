"""Check the CLI representations of regional masks and their XY registrations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io._cli.commands.phenocycler import phenocycler_command

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestMaskOptions:
    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (["--labels", '{"A":"x","A":"y"}'], "must be unique"),
            (["--labels", '{"A":1}'], "mask paths"),
            (["--labels-to-image", '{"A": [2,0,1]}'], "3-by-3"),
        ],
    )
    def test_invalid_mapping(self, tmp_path: Path, runner: CliRunner, arguments: list[str], message: str) -> None:
        result = runner.invoke(phenocycler_command, ["-i", str(tmp_path), "-o", str(tmp_path / "out"), *arguments])
        assert result.exit_code == 2
        assert message in result.output

    def test_single_file_and_affine(self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture) -> None:
        reader = mocker.patch("spatialdata_io.readers.phenocycler.phenocycler")
        result = runner.invoke(
            phenocycler_command,
            [
                "-i",
                str(tmp_path),
                "-o",
                str(tmp_path / "out"),
                "--labels",
                "mask.tif",
                "--labels-to-image",
                "[[1,0,3],[0,1,4],[0,0,1]]",
            ],
        )
        assert result.exit_code == 0, result.output
        assert reader.call_args.kwargs["labels"] == "mask.tif"
        assert reader.call_args.kwargs["labels_to_image"].matrix.tolist() == [[1, 0, 3], [0, 1, 4], [0, 0, 1]]
