"""Test the deliberately small shared CLI boundary."""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from spatialdata_io._cli._common import chunks_option, scale_factors_option
from spatialdata_io._raster_defaults import DEFAULT_CHUNKS


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
