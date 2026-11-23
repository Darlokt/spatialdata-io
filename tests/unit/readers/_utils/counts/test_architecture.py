"""Architecture tests for the typed 10x source boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SOURCES = tuple(Path("src/spatialdata_io/readers/_utils/counts") / name for name in ("_10x_h5.py", "_10x_mtx.py"))
_FORBIDDEN_IMPORT_ROOTS = {
    "dask",
    "geopandas",
    "scanpy",
    "shapely",
    "spatialdata",
    "xarray",
}
_FORBIDDEN_CALLS = {
    "compute",
    "exists",
    "is_dir",
    "load",
    "read_text",
    "resolve",
    "stat",
    "to_memory",
    "toarray",
    "todense",
}


class TestCountsArchitecture:
    @pytest.mark.parametrize("source", _SOURCES)
    def test_source_has_only_direct_source_representation_and_error_dependencies(self, source: Path) -> None:
        tree = ast.parse(source.read_text())
        imported_modules = {module for node in ast.walk(tree) for module in self._imported_modules(node)}
        roots = {module.split(".")[0] for module in imported_modules}

        assert roots.isdisjoint(_FORBIDDEN_IMPORT_ROOTS)
        assert "anndata.io" not in imported_modules
        assert not any("spatialdata_io.readers." in module and "._utils." not in module for module in imported_modules)
        assert "spatialdata_io.readers._utils.table._linkage" not in imported_modules

    @pytest.mark.parametrize("source", _SOURCES)
    def test_implementation_has_no_revalidation_computation_or_densification(self, source: Path) -> None:
        tree = ast.parse(source.read_text())
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

        assert calls.isdisjoint(_FORBIDDEN_CALLS)
        assert "A" not in attributes

    @staticmethod
    def _imported_modules(node: ast.AST) -> tuple[str, ...]:
        if isinstance(node, ast.Import):
            return tuple(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            return (node.module,)
        return ()
