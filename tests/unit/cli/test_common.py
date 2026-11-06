"""Test the deliberately small shared CLI boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click
import pytest
from click.testing import CliRunner

from spatialdata_io._cli._common import chunks_option, dataset_io_options, path_policy_option, scale_factors_option
from spatialdata_io._raster_defaults import DEFAULT_CHUNKS

if TYPE_CHECKING:
    from pathlib import Path


@click.command()
@scale_factors_option
@chunks_option
def _probe(scale_factors: tuple[int, ...], chunks: int | tuple[int, int, int]) -> None:
    click.echo(repr((scale_factors, chunks)))


class TestRasterOptions:
    """Shared pyramid and chunk options carry the central defaults to every raster command."""

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            ([], ((), DEFAULT_CHUNKS)),
            (["--scale-factors", "2", "--scale-factors", "4", "--chunks", "512"], ((2, 4), 512)),
            (["--chunks", "1", "--chunks", "256", "--chunks", "512"], ((), (1, 256, 512))),
        ],
        ids=["defaults", "factors-and-size", "cyx"],
    )
    def test_normalized_values(self, arguments: list[str], expected: tuple[object, object]) -> None:
        result = CliRunner().invoke(_probe, arguments)

        assert result.exit_code == 0, result.output
        assert result.output.strip() == repr(expected)

    @pytest.mark.parametrize(
        "arguments",
        [["--chunks", "4", "--chunks", "8"], ["--chunks", "0"], ["--scale-factors", "1"]],
        ids=["two-sizes", "zero", "factor-one"],
    )
    def test_invalid_values(self, arguments: list[str]) -> None:
        result = CliRunner().invoke(_probe, arguments)

        assert result.exit_code != 0

    def test_help_shows_the_chunk_default(self) -> None:
        result = CliRunner().invoke(_probe, ["--help"])

        assert "[default: 1, 1024, 1024; x>=1]" in " ".join(result.output.split())


@click.command()
@dataset_io_options
@path_policy_option
def _dataset_probe(input_path: object, output_path: object, path_policy: str) -> None:
    click.echo(repr((input_path, output_path, path_policy)))


class TestDatasetOptions:
    """The shared dataset options and the trailing symlink policy of the input root."""

    def test_dataset_options_are_input_and_output_only(self) -> None:
        assert [parameter.name for parameter in _dataset_probe.params] == ["input_path", "output_path", "path_policy"]

    @pytest.mark.parametrize(("arguments", "expected"), [([], "strict"), (["--path-policy", "relaxed"], "relaxed")])
    def test_path_policy_values(self, tmp_path: Path, arguments: list[str], expected: str) -> None:
        result = CliRunner().invoke(_dataset_probe, ["-i", str(tmp_path), "-o", str(tmp_path / "out"), *arguments])

        assert result.exit_code == 0, result.output
        assert result.output.strip().endswith(f"{expected!r})")

    def test_rejects_unknown_path_policy(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            _dataset_probe, ["-i", str(tmp_path), "-o", str(tmp_path / "out"), "--path-policy", "lenient"]
        )

        assert result.exit_code == 2
        assert "'lenient' is not one of 'strict', 'relaxed'" in result.output

    def test_help_shows_choices_and_default(self) -> None:
        result = CliRunner().invoke(_dataset_probe, ["--help"])
        help_text = " ".join(result.output.split())

        assert "--path-policy [strict|relaxed]" in help_text
        assert "git-annex or DVC" in help_text
        assert "[default: strict]" in help_text
