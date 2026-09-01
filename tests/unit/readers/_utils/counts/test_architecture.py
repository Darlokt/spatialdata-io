"""Architecture tests for the typed 10x source boundary."""

from __future__ import annotations

import ast
from pathlib import Path

_SOURCE = Path("src/spatialdata_io/readers/_utils/counts/_10x.py")
_FORBIDDEN_IMPORT_ROOTS = {
    "dask",
    "geopandas",
    "h5py",
    "shapely",
    "spatialdata",
    "xarray",
}
_FORBIDDEN_CALLS = {
    "compute",
    "exists",
    "is_dir",
    "is_file",
    "load",
    "read_text",
    "resolve",
    "stat",
    "to_memory",
    "toarray",
    "todense",
}


class TestCountsArchitecture:
    def test_source_has_only_direct_source_representation_and_error_dependencies(self) -> None:
        tree = ast.parse(_SOURCE.read_text())
        imported_modules = {module for node in ast.walk(tree) for module in self._imported_modules(node)}
        roots = {module.split(".")[0] for module in imported_modules}

        assert roots.isdisjoint(_FORBIDDEN_IMPORT_ROOTS)
        assert "scanpy.readwrite" not in imported_modules
        assert "anndata.io" not in imported_modules
        assert not any("spatialdata_io.readers." in module and "._utils." not in module for module in imported_modules)
        assert "spatialdata_io.readers._utils.table._linkage" not in imported_modules

    def test_implementation_has_no_revalidation_computation_or_densification(self) -> None:
        tree = ast.parse(_SOURCE.read_text())
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
