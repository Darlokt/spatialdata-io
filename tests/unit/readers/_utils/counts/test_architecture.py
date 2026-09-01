"""Architecture tests for the typed 10x source boundary."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from spatialdata_io.readers._utils.counts import read_10x_h5, read_10x_mtx

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
    "copy",
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

    def test_implementation_has_no_revalidation_computation_or_copy_calls(self) -> None:
        tree = ast.parse(_SOURCE.read_text())
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

        assert calls.isdisjoint(_FORBIDDEN_CALLS)
        assert "A" not in attributes

    def test_signatures_preserve_typed_artifact_and_explicit_option_contracts(self) -> None:
        h5_signature = inspect.signature(read_10x_h5)
        mtx_signature = inspect.signature(read_10x_mtx)
        h5_parameters = h5_signature.parameters
        mtx_parameters = mtx_signature.parameters

        assert set(h5_parameters) == {"path", "context", "genome", "gex_only"}
        assert set(mtx_parameters) == {
            "path",
            "context",
            "var_names",
            "make_unique",
            "gex_only",
            "prefix",
            "compressed",
        }
        assert all(parameter.kind is not inspect.Parameter.VAR_KEYWORD for parameter in h5_parameters.values())
        assert all(parameter.kind is not inspect.Parameter.VAR_KEYWORD for parameter in mtx_parameters.values())
        assert h5_parameters["path"].annotation == "ArtifactFile"
        assert mtx_parameters["path"].annotation == "ArtifactDirectory"
        assert h5_signature.return_annotation == "AnnData"
        assert mtx_signature.return_annotation == "AnnData"
        assert h5_parameters["path"].kind is inspect.Parameter.POSITIONAL_ONLY
        assert mtx_parameters["path"].kind is inspect.Parameter.POSITIONAL_ONLY
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for name, parameter in h5_parameters.items()
            if name != "path"
        )
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for name, parameter in mtx_parameters.items()
            if name != "path"
        )
        assert h5_parameters["context"].default is inspect.Parameter.empty
        assert mtx_parameters["context"].default is inspect.Parameter.empty
        assert h5_parameters["genome"].default is None
        assert h5_parameters["gex_only"].default is True
        assert mtx_parameters["var_names"].default == "gene_symbols"
        assert mtx_parameters["make_unique"].default is True
        assert mtx_parameters["gex_only"].default is True
        assert mtx_parameters["prefix"].default is None
        assert mtx_parameters["compressed"].default is True

    @staticmethod
    def _imported_modules(node: ast.AST) -> tuple[str, ...]:
        if isinstance(node, ast.Import):
            return tuple(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            return (node.module,)
        return ()
