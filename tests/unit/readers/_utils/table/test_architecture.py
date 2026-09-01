"""Architecture tests for the sparse table boundary."""

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN_ARRAY_IMPORTS = {
    "anndata",
    "dask",
    "pandas",
    "scanpy",
    "spatialdata",
    "spatialdata_io",
}
_FORBIDDEN_NORMALIZE_IMPORTS = _FORBIDDEN_ARRAY_IMPORTS - {"anndata", "spatialdata_io"}
_FORBIDDEN_LINKAGE_IMPORTS = {"dask", "geopandas", "shapely", "xarray"}
_FORBIDDEN_CALLS = {"compute", "load", "todense", "toarray", "to_memory"}
_NORMALIZE_INTERNAL_IMPORTS = {"spatialdata_io.readers._utils.table._arrays"}


class TestTableArchitecture:
    """Protect the model-independent, non-densifying package boundary."""

    def test_arrays_remain_model_independent(self) -> None:
        path = Path("src/spatialdata_io/readers/_utils/table/_arrays.py")
        tree = ast.parse(path.read_text())
        imports = {module.split(".")[0] for node in ast.walk(tree) for module in self._imported_modules(node)}
        assert imports.isdisjoint(_FORBIDDEN_ARRAY_IMPORTS)

    def test_normalization_has_only_the_anndata_model_dependency(self) -> None:
        path = Path("src/spatialdata_io/readers/_utils/table/_normalize.py")
        tree = ast.parse(path.read_text())
        imported_modules = {module for node in ast.walk(tree) for module in self._imported_modules(node)}
        import_roots = {module.split(".")[0] for module in imported_modules}
        internal_imports = {module for module in imported_modules if module.startswith("spatialdata_io.")}
        assert import_roots.isdisjoint(_FORBIDDEN_NORMALIZE_IMPORTS)
        assert internal_imports <= _NORMALIZE_INTERNAL_IMPORTS

    def test_has_no_lazy_or_densifying_calls(self) -> None:
        package = Path("src/spatialdata_io/readers/_utils/table")
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text())
            calls = {
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
            assert calls.isdisjoint(_FORBIDDEN_CALLS), path
            assert "A" not in attributes, path

    def test_linkage_is_model_adapter_without_spatial_element_dependencies(self) -> None:
        path = Path("src/spatialdata_io/readers/_utils/table/_linkage.py")
        tree = ast.parse(path.read_text())
        imported_modules = {module for node in ast.walk(tree) for module in self._imported_modules(node)}
        import_roots = {module.split(".")[0] for module in imported_modules}
        assert import_roots.isdisjoint(_FORBIDDEN_LINKAGE_IMPORTS)

    @staticmethod
    def _imported_modules(node: ast.AST) -> tuple[str, ...]:
        if isinstance(node, ast.Import):
            return tuple(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            return (node.module,)
        return ()
