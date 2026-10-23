"""Verify Curio package and lazy root exports."""

from __future__ import annotations

import importlib

from spatialdata_io import curio as root_curio
from spatialdata_io.readers.curio import curio
from spatialdata_io.readers.curio._reader import curio as reader_curio


class TestPackage:
    """Keep one intentional public entry point at the established import paths."""

    def test_public_exports(self) -> None:
        package = importlib.import_module("spatialdata_io.readers.curio")

        assert package.__all__ == ["curio"]
        assert package.curio is curio
        assert curio is reader_curio
        assert root_curio is curio
