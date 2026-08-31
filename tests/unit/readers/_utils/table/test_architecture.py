"""Architecture tests for the sparse table boundary."""

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN_IMPORTS = {
    "anndata",
    "dask",
    "pandas",
    "scanpy",
    "spatialdata",
}
_FORBIDDEN_CALLS = {"compute", "todense", "toarray"}


class TestTableArchitecture:
    """Protect the model-independent, non-densifying package boundary."""

    def test_has_no_model_lazy_or_densifying_dependencies(self) -> None:
        package = Path("src/spatialdata_io/readers/_utils/table")
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text())
            imports = {module.split(".")[0] for node in ast.walk(tree) for module in self._imported_modules(node)}
            calls = {
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
            assert imports.isdisjoint(_FORBIDDEN_IMPORTS), path
            assert calls.isdisjoint(_FORBIDDEN_CALLS), path
            assert "A" not in attributes, path

    @staticmethod
    def _imported_modules(node: ast.AST) -> tuple[str, ...]:
        if isinstance(node, ast.Import):
            return tuple(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            return (node.module,)
        return ()
