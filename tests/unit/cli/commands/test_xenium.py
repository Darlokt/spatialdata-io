"""Test the Xenium command boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata_io._cli.commands.xenium import xenium_command

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestXeniumCommand:
    """Verify boolean and legacy JSON conversion."""

    def test_forwards_options_and_writes_output(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.xenium.xenium", return_value=reader_result)
        output = tmp_path / "output.zarr"
        false_options = (
            "cells-boundaries",
            "nucleus-boundaries",
            "cells-labels",
            "nucleus-labels",
            "transcripts",
            "morphology-mip",
            "morphology-focus",
            "aligned-images",
            "cells-table",
            "gex-only",
        )
        arguments = ["--input", str(tmp_path), "--output", str(output)]
        for option in false_options:
            arguments.extend((f"--{option}", "false"))
        arguments.extend(
            (
                "--cells-as-circles",
                "true",
                "--imread-kwargs",
                '{"n": 2}',
                "--image-models-kwargs",
                '{"scale_factors": null}',
                "--labels-models-kwargs",
                '{"scale_factors": null}',
            )
        )

        result = runner.invoke(xenium_command, arguments)

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            cells_boundaries=False,
            nucleus_boundaries=False,
            cells_as_circles=True,
            cells_labels=False,
            nucleus_labels=False,
            transcripts=False,
            morphology_mip=False,
            morphology_focus=False,
            aligned_images=False,
            cells_table=False,
            gex_only=False,
            imread_kwargs={"n": 2},
            image_models_kwargs={"scale_factors": None},
            labels_models_kwargs={"scale_factors": None},
        )
        reader_result.write.assert_called_once_with(output)
