from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN_IMPORTS = {
    "PIL",
    "dask_image",
    "imageio",
    "rasterio",
    "rioxarray",
    "spatialdata",
}
_FORBIDDEN_CALLS = {"compute", "load", "memmap"}


class TestImageArchitecture:
    """Protect the model-independent, lazy centralized boundary."""

    def test_has_no_model_or_legacy_loader_dependencies(self) -> None:
        package = Path("src/spatialdata_io/readers/_utils/image")
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text())
            imports = {module.split(".")[0] for node in ast.walk(tree) for module in self._imported_modules(node)}
            calls = {
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            assert imports.isdisjoint(_FORBIDDEN_IMPORTS), path
            assert calls.isdisjoint(_FORBIDDEN_CALLS), path

    @staticmethod
    def _imported_modules(node: ast.AST) -> tuple[str, ...]:
        if isinstance(node, ast.Import):
            return tuple(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            return (node.module,)
        return ()
