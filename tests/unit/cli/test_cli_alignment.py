"""Verify reader and extracted Click command alignment."""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import pytest

from spatialdata_io._cli import cli
from spatialdata_io._cli.commands.generic import generic_command
from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS


@dataclass(frozen=True, slots=True)
class _ReaderCase:
    module: str
    reader: str
    command: str


_READER_CASES = (
    _ReaderCase("spatialdata_io.readers.codex", "codex", "codex"),
    pytest.param(_ReaderCase("spatialdata_io.readers.cosmx", "cosmx", "cosmx"), marks=pytest.mark.cosmx),
    pytest.param(_ReaderCase("spatialdata_io.readers.curio", "curio", "curio"), marks=pytest.mark.curio),
    _ReaderCase("spatialdata_io.readers.dbit", "dbit", "dbit"),
    _ReaderCase("spatialdata_io.readers.iss", "iss", "iss"),
    pytest.param(_ReaderCase("spatialdata_io.readers.macsima", "macsima", "macsima"), marks=pytest.mark.macsima),
    _ReaderCase("spatialdata_io.readers.mcmicro", "mcmicro", "mcmicro"),
    pytest.param(_ReaderCase("spatialdata_io.readers.merscope", "merscope", "merscope"), marks=pytest.mark.merscope),
    _ReaderCase("spatialdata_io.readers.seqfish", "seqfish", "seqfish"),
    _ReaderCase("spatialdata_io.readers.steinbock", "steinbock", "steinbock"),
    _ReaderCase("spatialdata_io.readers.stereoseq", "stereoseq", "stereoseq"),
    _ReaderCase("spatialdata_io.readers.visium", "visium", "visium"),
    _ReaderCase("spatialdata_io.readers.visium_hd", "visium_hd", "visium-hd"),
    _ReaderCase("spatialdata_io.readers.xenium", "xenium", "xenium"),
)
# Readers without images or labels.
_WITHOUT_RASTER_OPTIONS = frozenset({"curio"})
_RASTER_DEFAULTS = {"scale_factors": DEFAULT_SCALE_FACTORS, "chunks": DEFAULT_CHUNKS}


def _reader_parameters(case: _ReaderCase) -> set[str]:
    module = importlib.import_module(case.module)
    reader = getattr(module, case.reader)
    return {
        name
        for name, parameter in inspect.signature(reader).parameters.items()
        if name != "path" and parameter.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }


def _command_parameters(case: _ReaderCase) -> set[str]:
    command = cli.commands[case.command]
    return {parameter.name for parameter in command.params if parameter.name not in {"input_path", "output_path"}}


class TestCliAlignment:
    """Check stable external command registration and current option coverage."""

    @pytest.mark.parametrize("case", _READER_CASES, ids=lambda case: case.reader)
    def test_vendor_command_and_reader_parameter_names_agree(self, case: _ReaderCase) -> None:
        assert _reader_parameters(case) == _command_parameters(case)

    @pytest.mark.parametrize("case", _READER_CASES, ids=lambda case: case.reader)
    def test_raster_options_use_the_central_defaults(self, case: _ReaderCase) -> None:
        reader = getattr(importlib.import_module(case.module), case.reader)
        parameters = inspect.signature(reader).parameters
        command = {parameter.name: parameter for parameter in cli.commands[case.command].params}

        if case.reader in _WITHOUT_RASTER_OPTIONS:
            assert not _RASTER_DEFAULTS.keys() & parameters.keys()
            return
        for name, default in _RASTER_DEFAULTS.items():
            assert parameters[name].default == default
            assert command[name].default == default


@pytest.mark.generic
class TestGenericCommandContract:
    """Check the fully migrated Generic command boundary."""

    def test_reader_cli_conjunction_collects_command_and_workflow_tests(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "--no-cov",
                "-q",
                "-m",
                "generic and cli",
                "tests/unit/cli/commands/test_generic.py",
                "tests/integration/readers/generic",
            ],
            cwd=Path(__file__).parents[3],
            check=True,
            capture_output=True,
            text=True,
        )

        assert "TestGenericCommand::test_forwards_options_and_reports_new_output" in completed.stdout
        assert "TestReadGenericWrapper::test_writes_selected_tiff_series" in completed.stdout
        assert "TestImage::" not in completed.stdout

    def test_uses_typed_paths_and_exact_chunk_cardinality(self) -> None:
        options = {parameter.name: parameter for parameter in generic_command.params}

        assert isinstance(options["input_path"].type, click.Path)
        assert options["input_path"].type.type is Path
        assert options["input_path"].required
        assert isinstance(options["output_path"].type, click.Path)
        assert options["output_path"].type.type is Path
        assert options["output_path"].required
        assert options["chunks"].multiple
        assert options["chunks"].default == ()
        assert options["scale_factors"].multiple
        assert options["scale_factors"].default == ()
        assert options["tiff_series"].default == 0
        assert options["tiff_level"].default == 0
