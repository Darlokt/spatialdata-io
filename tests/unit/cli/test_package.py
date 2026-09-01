"""Test command registration and import boundaries."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import click
from click.testing import CliRunner

from spatialdata_io._cli import cli

_COMMANDS = {
    "codex",
    "cosmx",
    "curio",
    "dbit",
    "generic",
    "iss",
    "macsima",
    "mcmicro",
    "merscope",
    "seqfish",
    "steinbock",
    "stereoseq",
    "visium",
    "visium-hd",
    "xenium",
}


class TestCliPackage:
    """Test the cheap top-level command surface."""

    def test_registers_every_command(self) -> None:
        assert set(cli.commands) == _COMMANDS

    def test_commands_share_typed_input_and_output_paths(self) -> None:
        for name, command in cli.commands.items():
            options = {parameter.name: parameter for parameter in command.params}
            input_type = options["input_path"].type
            output_type = options["output_path"].type
            assert isinstance(input_type, click.Path)
            assert isinstance(output_type, click.Path)
            assert input_type.type is Path
            assert output_type.type is Path
            assert output_type.file_okay is False
            if name == "generic":
                assert input_type.file_okay is True
                assert input_type.dir_okay is False
            else:
                assert input_type.file_okay is False
                assert input_type.dir_okay is True

    def test_command_modules_do_not_import_readers_at_module_scope(self) -> None:
        commands_directory = Path(__file__).parents[3] / "src" / "spatialdata_io" / "_cli" / "commands"

        for path in commands_directory.glob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in tree.body:
                if isinstance(node, ast.Import):
                    assert not any(alias.name.startswith("spatialdata_io.readers") for alias in node.names), path.name
                elif isinstance(node, ast.ImportFrom):
                    assert node.module is None or not node.module.startswith("spatialdata_io.readers"), path.name

    def test_vendor_options_have_help_and_visible_defaults(self) -> None:
        for name, command in cli.commands.items():
            visible_defaults = 0
            for parameter in command.params:
                assert isinstance(parameter, click.Option)
                assert parameter.help, f"{name} --{parameter.name} has no help text"
                if name != "generic" and parameter.default is not None and not parameter.required:
                    assert parameter.show_default, f"{name} --{parameter.name} hides its default"
                    visible_defaults += 1

            if name != "generic":
                help_result = CliRunner().invoke(command, ["--help"])
                assert help_result.exit_code == 0
                assert help_result.output.count("[default:") == visible_defaults

    def test_help_imports_no_readers_or_heavy_scientific_packages(self, tmp_path: Path) -> None:
        code = """
import json
import sys
from click.testing import CliRunner
from spatialdata_io._cli import cli

result = CliRunner().invoke(cli, ["--help"])
if result.exit_code != 0:
    raise SystemExit(result.output)
reader_modules = sorted(name for name in sys.modules if name.startswith("spatialdata_io.readers"))
heavy_packages = sorted(
    {
        name.split(".")[0]
        for name in sys.modules
        if name.split(".")[0]
        in {"anndata", "dask", "geopandas", "numpy", "pandas", "scipy", "spatialdata", "tifffile", "xarray"}
    }
)
print(json.dumps({"readers": reader_modules, "heavy": heavy_packages}))
"""

        completed = subprocess.run(  # noqa: S603, RUF100 - executable and script are test-controlled.
            [sys.executable, "-c", code],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

        assert json.loads(completed.stdout) == {"readers": [], "heavy": []}
