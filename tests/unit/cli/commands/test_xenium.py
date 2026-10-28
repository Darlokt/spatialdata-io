"""Exact Xenium CLI selections, polarity, cardinality and output ownership."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io._cli.commands.xenium import FEATURE_TYPES, xenium_command
from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from click.testing import CliRunner
    from pytest_mock import MockerFixture

DEFAULTS: dict[str, object] = {
    "include_table": True,
    "feature_types": None,
    "table_target": "cell_labels",
    "include_cell_labels": True,
    "include_nucleus_labels": True,
    "include_cell_boundaries": True,
    "cell_geometry": "polygons",
    "include_nucleus_boundaries": True,
    "invalid_geometries": "keep",
    "include_transcripts": True,
    "include_morphology_focus": True,
    "include_morphology_mip": False,
    "include_aligned_images": True,
    "include_analysis": False,
    "scale_factors": DEFAULT_SCALE_FACTORS,
    "chunks": DEFAULT_CHUNKS,
}


class TestXeniumCommand:
    """Keep explicit product selection aligned with the reader signature."""

    def test_defaults_match_the_reader(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.xenium.xenium", return_value=reader_result)
        output = tmp_path / "output.zarr"

        result = runner.invoke(xenium_command, ["-i", str(tmp_path), "-o", str(output)])

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(tmp_path, **DEFAULTS)
        reader_result.write.assert_called_once_with(output)

    def test_forwards_every_option(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, reader_result: MagicMock
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.xenium.xenium", return_value=reader_result)
        arguments = [
            "-i",
            str(tmp_path),
            "-o",
            str(tmp_path / "output.zarr"),
            "--no-include-table",
            "--feature-types",
            "Protein Expression",
            "--feature-types",
            "Gene Expression",
            "--table-target",
            "none",
            "--no-include-cell-labels",
            "--no-include-nucleus-labels",
            "--no-include-cell-boundaries",
            "--cell-geometry",
            "both",
            "--no-include-nucleus-boundaries",
            "--invalid-geometries",
            "drop",
            "--no-include-transcripts",
            "--no-include-morphology-focus",
            "--include-morphology-mip",
            "--no-include-aligned-images",
            "--include-analysis",
            "--scale-factors",
            "2",
            "--scale-factors",
            "4",
            "--chunks",
            "1",
            "--chunks",
            "512",
            "--chunks",
            "256",
        ]

        result = runner.invoke(xenium_command, arguments)

        assert result.exit_code == 0, result.output
        reader.assert_called_once_with(
            tmp_path,
            include_table=False,
            feature_types=("Protein Expression", "Gene Expression"),
            table_target=None,
            include_cell_labels=False,
            include_nucleus_labels=False,
            include_cell_boundaries=False,
            cell_geometry="both",
            include_nucleus_boundaries=False,
            invalid_geometries="drop",
            include_transcripts=False,
            include_morphology_focus=False,
            include_morphology_mip=True,
            include_aligned_images=False,
            include_analysis=True,
            scale_factors=(2, 4),
            chunks=(1, 512, 256),
        )

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            (["--feature-types", "all"], {"feature_types": "all"}),
            (["--chunks", "128"], {"chunks": 128}),
            (["--table-target", "cell_circles", "--cell-geometry", "circles"], {"table_target": "cell_circles"}),
        ],
        ids=["all-feature-types", "isotropic-chunks", "circles"],
    )
    def test_normalized_values(
        self,
        tmp_path: Path,
        runner: CliRunner,
        mocker: MockerFixture,
        arguments: list[str],
        expected: dict[str, object],
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.xenium.xenium")

        result = runner.invoke(xenium_command, ["-i", str(tmp_path), "-o", str(tmp_path / "out.zarr"), *arguments])

        assert result.exit_code == 0, result.output
        options = reader.call_args.kwargs
        assert {name: options[name] for name in expected} == expected

    @pytest.mark.parametrize(
        "arguments",
        [
            ["--feature-types", "all", "--feature-types", "Gene Expression"],
            ["--feature-types", "Gene Expression", "--feature-types", "Gene Expression"],
            ["--feature-types", "Gene"],
            ["--table-target", "nucleus_labels"],
            ["--cell-geometry", "squares"],
            ["--invalid-geometries", "repair"],
            ["--scale-factors", "1"],
            ["--chunks", "0"],
            ["--chunks", "4", "--chunks", "8"],
            ["--cells-as-circles", "true"],
            ["--imread-kwargs", "{}"],
        ],
        ids=[
            "all-with-types",
            "repeated-type",
            "unknown-type",
            "target",
            "geometry",
            "policy",
            "scale-factor",
            "chunk-size",
            "chunk-count",
            "removed-circles",
            "removed-kwargs",
        ],
    )
    def test_invalid_options_do_not_call_reader(
        self, tmp_path: Path, runner: CliRunner, mocker: MockerFixture, arguments: list[str]
    ) -> None:
        reader = mocker.patch("spatialdata_io.readers.xenium.xenium")

        result = runner.invoke(xenium_command, ["-i", str(tmp_path), "-o", str(tmp_path / "out.zarr"), *arguments])

        assert result.exit_code != 0
        reader.assert_not_called()

    def test_option_defaults_and_choices(self) -> None:
        options = {parameter.name: parameter for parameter in xenium_command.params}

        for name in ("feature_types", "scale_factors", "chunks"):
            assert options[name].multiple
        assert options["feature_types"].default == ()
        assert options["scale_factors"].default == DEFAULT_SCALE_FACTORS
        assert options["chunks"].default == DEFAULT_CHUNKS
        assert set(FEATURE_TYPES) | {"all"} == set(getattr(options["feature_types"].type, "choices", ()))
        assert getattr(options["table_target"].type, "choices", ()) == (
            "cell_labels",
            "cell_boundaries",
            "cell_circles",
            "none",
        )
