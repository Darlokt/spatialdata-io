"""CODEX package exports and public transaction ownership."""

from __future__ import annotations

import importlib
import inspect

from spatialdata_io import codex as root_codex
from spatialdata_io.readers.codex import codex


def test_package_and_lazy_root_export() -> None:
    """Retain only the stable reader through both intentional public imports."""
    package = importlib.import_module("spatialdata_io.readers.codex")

    assert package.__all__ == ["codex"]
    assert package.codex is root_codex is codex


class TestCodex:
    """Keep all supported optional arguments explicit and keyword-only."""

    def test_signature_is_explicit(self) -> None:
        parameters = inspect.signature(codex).parameters

        assert list(parameters) == ["path", "cell_table_file", "image_file", "data_axes", "tiff_series", "chunks"]
        assert all(
            parameter.kind == inspect.Parameter.KEYWORD_ONLY for name, parameter in parameters.items() if name != "path"
        )
