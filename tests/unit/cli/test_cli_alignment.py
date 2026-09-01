"""Verify reader and extracted Click command alignment."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path

import click
import pytest

from spatialdata_io._cli import cli
from spatialdata_io._cli.commands.generic import generic_command


@dataclass(frozen=True, slots=True)
class _ReaderCase:
    module: str
    reader: str
    command: str


_READER_CASES = (
    _ReaderCase("spatialdata_io.readers.codex", "codex", "codex"),
    _ReaderCase("spatialdata_io.readers.cosmx", "cosmx", "cosmx"),
    _ReaderCase("spatialdata_io.readers.curio", "curio", "curio"),
    _ReaderCase("spatialdata_io.readers.dbit", "dbit", "dbit"),
    _ReaderCase("spatialdata_io.readers.iss", "iss", "iss"),
    _ReaderCase("spatialdata_io.readers.macsima", "macsima", "macsima"),
    _ReaderCase("spatialdata_io.readers.mcmicro", "mcmicro", "mcmicro"),
    _ReaderCase("spatialdata_io.readers.merscope", "merscope", "merscope"),
    _ReaderCase("spatialdata_io.readers.seqfish", "seqfish", "seqfish"),
    _ReaderCase("spatialdata_io.readers.steinbock", "steinbock", "steinbock"),
    _ReaderCase("spatialdata_io.readers.stereoseq", "stereoseq", "stereoseq"),
    _ReaderCase("spatialdata_io.readers.visium", "visium", "visium"),
    _ReaderCase("spatialdata_io.readers.visium_hd", "visium_hd", "visium-hd"),
    _ReaderCase("spatialdata_io.readers.xenium", "xenium", "xenium"),
)

_READER_EXCEPTIONS = {"xenium": frozenset(("n_jobs",))}


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
        exceptions = _READER_EXCEPTIONS.get(case.reader, frozenset())
        assert _reader_parameters(case) - exceptions == _command_parameters(case)


class TestGenericCommandContract:
    """Check the fully migrated Generic command boundary."""

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
        assert options["tiff_series"].default == 0
        assert options["tiff_level"].default == 0
