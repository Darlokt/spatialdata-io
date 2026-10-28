"""Intentional Steinbock package exports and lazy root import contract."""

from __future__ import annotations

import importlib
import inspect

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS


class TestPackage:
    """Preserve the established public reader path while exporting only its entry point."""

    def test_public_entry_point_and_root_identity(self) -> None:
        package = importlib.import_module("spatialdata_io.readers.steinbock")
        implementation = importlib.import_module("spatialdata_io.readers.steinbock._reader")
        root = importlib.import_module("spatialdata_io")

        assert package.__all__ == ["steinbock"]
        assert package.steinbock is implementation.steinbock is root.steinbock

    def test_public_signature_keeps_explicit_keyword_only_options(self) -> None:
        package = importlib.import_module("spatialdata_io.readers.steinbock")
        parameters = inspect.signature(package.steinbock).parameters

        assert list(parameters) == ["path", "labels_kind", "scale_factors", "chunks"]
        assert parameters["labels_kind"].default == "deepcell"
        assert all(
            parameter.kind == inspect.Parameter.KEYWORD_ONLY for name, parameter in parameters.items() if name != "path"
        )
        assert parameters["scale_factors"].default == DEFAULT_SCALE_FACTORS
        assert parameters["chunks"].default == DEFAULT_CHUNKS
