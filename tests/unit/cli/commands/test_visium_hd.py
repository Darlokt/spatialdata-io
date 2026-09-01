"""Test the Visium HD command boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata_io._cli.commands.visium_hd import visium_hd_command

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture


class TestVisiumHdCommand:
    """Verify repeated, tri-state, and legacy JSON conversion."""

    def test_forwards_options_and_writes_output(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.visium_hd.visium_hd", return_value=reader_result)
        image = tmp_path / "image.tif"
        image.touch()
        output = tmp_path / "output.zarr"

        result = runner.invoke(
            visium_hd_command,
            [
                "--input",
                str(tmp_path),
                "--output",
                str(output),
                "--dataset-id",
                "sample",
                "--filtered-counts-file",
                "false",
                "--bin-size",
                "2",
                "--bin-size",
                "8",
                "--bins-as-squares",
                "false",
                "--fullres-image-file",
                str(image),
                "--load-all-images",
                "true",
                "--annotate-table-by-labels",
                "true",
                "--load-segmentations-only",
                "true",
                "--load-nucleus-segmentations",
                "true",
                "--var-names-make-unique",
                "false",
                "--gex-only",
                "true",
                "--imread-kwargs",
                '{"n": 2}',
                "--image-models-kwargs",
                '{"scale_factors": null}',
                "--anndata-kwargs",
                '{"gex_only": true}',
            ],
        )

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            path=tmp_path,
            dataset_id="sample",
            filtered_counts_file=False,
            load_segmentations_only=True,
            load_nucleus_segmentations=True,
            bin_size=[2, 8],
            bins_as_squares=False,
            fullres_image_file=image,
            load_all_images=True,
            annotate_table_by_labels=True,
            var_names_make_unique=False,
            gex_only=True,
            imread_kwargs={"n": 2},
            image_models_kwargs={"scale_factors": None},
            anndata_kwargs={"gex_only": True},
        )
        reader_result.write.assert_called_once_with(output)
