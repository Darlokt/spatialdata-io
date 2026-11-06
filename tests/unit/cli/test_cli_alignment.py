"""Verify reader and extracted Click command alignment."""

from __future__ import annotations

import ast
import importlib
import inspect
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click
import pytest
from click.testing import CliRunner

from spatialdata_io._cli import cli
from spatialdata_io._cli.commands.generic import generic_command
from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@dataclass(frozen=True, slots=True)
class _ReaderCase:
    module: str
    reader: str
    command: str


_READER_CASES = (
    _ReaderCase("spatialdata_io.readers.codex", "codex", "codex"),
    pytest.param(_ReaderCase("spatialdata_io.readers.cosmx", "cosmx", "cosmx"), marks=pytest.mark.cosmx),
    pytest.param(_ReaderCase("spatialdata_io.readers.curio", "curio", "curio"), marks=pytest.mark.curio),
    pytest.param(_ReaderCase("spatialdata_io.readers.dbit", "dbit", "dbit"), marks=pytest.mark.dbit),
    pytest.param(_ReaderCase("spatialdata_io.experimental.iss", "iss", "iss"), marks=pytest.mark.iss),
    pytest.param(_ReaderCase("spatialdata_io.readers.macsima", "macsima", "macsima"), marks=pytest.mark.macsima),
    _ReaderCase("spatialdata_io.readers.mcmicro", "mcmicro", "mcmicro"),
    pytest.param(_ReaderCase("spatialdata_io.readers.merscope", "merscope", "merscope"), marks=pytest.mark.merscope),
    pytest.param(_ReaderCase("spatialdata_io.readers.seqfish", "seqfish", "seqfish"), marks=pytest.mark.seqfish),
    pytest.param(
        _ReaderCase("spatialdata_io.readers.steinbock", "steinbock", "steinbock"), marks=pytest.mark.steinbock
    ),
    _ReaderCase("spatialdata_io.readers.stereoseq", "stereoseq", "stereoseq"),
    _ReaderCase("spatialdata_io.readers.visium", "visium", "visium"),
    _ReaderCase("spatialdata_io.readers.visium_hd", "visium_hd", "visium-hd"),
    _ReaderCase("spatialdata_io.readers.xenium", "xenium", "xenium"),
)
# Readers without images or labels.
_WITHOUT_RASTER_OPTIONS = frozenset({"curio"})
_RASTER_DEFAULTS = {"scale_factors": DEFAULT_SCALE_FACTORS, "chunks": DEFAULT_CHUNKS}
_COMMAND_DIRECTORY = Path(__file__).parents[3] / "src" / "spatialdata_io" / "_cli" / "commands"
# Dataset command modules whose options end with ``--path-policy``; generic takes a single file.
_COMMAND_MODULES = {
    path.stem: path for path in _COMMAND_DIRECTORY.glob("*.py") if path.stem not in {"__init__", "generic", "curio"}
}
# Options that a command requires besides the dataset root and output store.
_REQUIRED_ARGUMENTS = {
    "iss": ("--raw-image-file", "raw.tif", "--labels-file", "labels.tif", "--table-file", "table.h5ad"),
}


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


class TestPathPolicyAlignment:
    """Every dataset command exposes the reader's symlink policy with the same default."""

    @pytest.mark.parametrize("case", _READER_CASES, ids=lambda case: case.reader)
    def test_reader_and_command_default_to_strict(self, case: _ReaderCase) -> None:
        reader = getattr(importlib.import_module(case.module), case.reader)
        parameter = inspect.signature(reader).parameters["path_policy"]
        option = {parameter.name: parameter for parameter in cli.commands[case.command].params}["path_policy"]

        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default == "strict"
        assert option.default == "strict"
        assert isinstance(option.type, click.Choice)
        assert list(option.type.choices) == ["strict", "relaxed"]

    @pytest.mark.parametrize("case", _READER_CASES, ids=lambda case: case.reader)
    def test_path_policy_is_the_last_reader_parameter_and_command_option(self, case: _ReaderCase) -> None:
        reader = getattr(importlib.import_module(case.module), case.reader)
        command = cli.commands[case.command]

        assert list(inspect.signature(reader).parameters)[-1] == "path_policy"
        assert command.params[-1].name == "path_policy"
        assert [parameter.name for parameter in command.params[:2]] == ["input_path", "output_path"]

    @pytest.mark.parametrize("command", sorted(_COMMAND_MODULES), ids=str)
    def test_option_typed_dicts_keep_their_docstring_first_and_path_policy_last(self, command: str) -> None:
        tree = ast.parse(_COMMAND_MODULES[command].read_text(encoding="utf-8"))
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name.endswith("Options")]

        assert len(classes) == 1
        for node in classes:
            # A docstring, when present, is the first statement; no string is stranded among the fields.
            assert all(isinstance(statement, ast.AnnAssign) for statement in node.body[1:])
            last = node.body[-1]
            assert isinstance(last, ast.AnnAssign)
            assert isinstance(last.target, ast.Name)
            assert last.target.id == "path_policy"

    @pytest.mark.parametrize("case", _READER_CASES, ids=lambda case: case.reader)
    def test_relaxed_flag_reaches_the_reader(self, case: _ReaderCase, tmp_path: Path, mocker: MockerFixture) -> None:
        reader = mocker.patch(f"{case.module}.{case.reader}")
        arguments = [
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path / "output.zarr"),
            "--path-policy",
            "relaxed",
            *_REQUIRED_ARGUMENTS.get(case.reader, ()),
        ]

        result = CliRunner().invoke(cli.commands[case.command], arguments)

        assert result.exit_code == 0, result.output
        reader.assert_called_once()
        assert reader.call_args.kwargs["path_policy"] == "relaxed"


@pytest.mark.dbit
class TestDbitCommandContract:
    """Keep DBiT CLI selection inclusive of command, alignment and persistence."""

    def test_reader_cli_conjunction_collects_command_alignment_and_workflow_tests(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "--no-cov",
                "-q",
                "-m",
                "dbit and cli",
                "tests/unit/cli/commands/test_dbit.py",
                "tests/unit/cli/test_cli_alignment.py",
                "tests/integration/readers/dbit",
            ],
            cwd=Path(__file__).parents[3],
            check=True,
            capture_output=True,
            text=True,
        )

        assert "TestDbitCommand::test_defaults_and_output_writing" in completed.stdout
        assert "TestCliAlignment::test_vendor_command_and_reader_parameter_names_agree[dbit]" in completed.stdout
        assert "TestCliAlignment::test_raster_options_use_the_central_defaults[dbit]" in completed.stdout
        assert "TestDbitCommandWorkflow::test_writes_linked_spatialdata_zarr" in completed.stdout
        assert "TestDbit::" not in completed.stdout
        assert "test_reader.py::" not in completed.stdout
        assert "test_grid.py::" not in completed.stdout


@pytest.mark.iss
class TestIssCommandContract:
    """Keep the ISS CLI selector inclusive of its complete public workflow."""

    def test_reader_cli_conjunction_collects_command_alignment_and_workflow_tests(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "--no-cov",
                "-q",
                "-m",
                "iss and cli",
                "tests/unit/cli/commands/test_iss.py",
                "tests/unit/cli/test_cli_alignment.py",
                "tests/integration/experimental/iss",
            ],
            cwd=Path(__file__).parents[3],
            check=True,
            capture_output=True,
            text=True,
        )

        assert "TestIssCommand::test_defaults_resolve_artifacts_in_reader" in completed.stdout
        assert "TestCliAlignment::test_vendor_command_and_reader_parameter_names_agree[iss]" in completed.stdout
        assert "TestCliAlignment::test_raster_options_use_the_central_defaults[iss]" in completed.stdout
        assert "TestIssCommandWorkflow::test_writes_linked_spatialdata_zarr" in completed.stdout
        assert "TestIss::" not in completed.stdout


@pytest.mark.steinbock
class TestSteinbockCommandContract:
    """Keep the Steinbock CLI selector inclusive of its complete public workflow."""

    def test_reader_cli_conjunction_collects_command_alignment_and_workflow_tests(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "--no-cov",
                "-q",
                "-m",
                "steinbock and cli",
                "tests/unit/cli/commands/test_steinbock.py",
                "tests/unit/cli/test_cli_alignment.py",
                "tests/integration/readers/steinbock",
            ],
            cwd=Path(__file__).parents[3],
            check=True,
            capture_output=True,
            text=True,
        )

        assert "TestSteinbockCommand::test_defaults_and_output_writing" in completed.stdout
        assert "TestCliAlignment::test_vendor_command_and_reader_parameter_names_agree[steinbock]" in completed.stdout
        assert "TestCliAlignment::test_raster_options_use_the_central_defaults[steinbock]" in completed.stdout
        assert "TestSteinbockCommandWorkflow::test_writes_linked_spatialdata_zarr" in completed.stdout
        assert "TestSteinbock::" not in completed.stdout
        assert "test_reader.py::" not in completed.stdout
        assert "test_positions.py::" not in completed.stdout


@pytest.mark.seqfish
class TestSeqfishCommandContract:
    """Keep seqFISH CLI selection inclusive of command, alignment and persistence."""

    def test_reader_cli_conjunction_collects_command_alignment_and_workflow_tests(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "--no-cov",
                "-q",
                "-m",
                "seqfish and cli",
                "tests/unit/cli/commands/test_seqfish.py",
                "tests/unit/cli/test_cli_alignment.py",
                "tests/integration/readers/seqfish",
            ],
            cwd=Path(__file__).parents[3],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "TestSeqfishCommand::test_defaults_forward_and_write" in completed.stdout
        assert "TestCliAlignment::test_vendor_command_and_reader_parameter_names_agree[seqfish]" in completed.stdout
        assert "TestCliAlignment::test_raster_options_use_the_central_defaults[seqfish]" in completed.stdout
        assert "test_seqfish.py::TestSeqfishCommand::test_complete_conversion" in completed.stdout
        assert "test_seqfish.py::TestSeqfishCommand::test_table_only_conversion" in completed.stdout
        assert "test_reader.py::" not in completed.stdout
        assert "test_characterization.py::" not in completed.stdout


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
        assert options["tiff_series"].default is None
        assert options["tiff_level"].default == 0
