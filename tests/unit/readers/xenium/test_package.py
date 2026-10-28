"""Intentional Xenium package exports and lazy root import behavior."""

from __future__ import annotations

import importlib
import subprocess
import sys


class TestPackage:
    """Keep the three readers public while layout, storage and decoding stay private."""

    def test_only_readers_are_exported(self) -> None:
        package = importlib.import_module("spatialdata_io.readers.xenium")
        reader = importlib.import_module("spatialdata_io.readers.xenium._reader")
        aligned = importlib.import_module("spatialdata_io.readers.xenium._aligned")
        selection = importlib.import_module("spatialdata_io.readers.xenium._selection")

        assert package.__all__ == ["xenium", "xenium_aligned_image", "xenium_explorer_selection"]
        assert package.xenium is reader.xenium
        assert package.xenium_aligned_image is aligned.xenium_aligned_image
        assert package.xenium_explorer_selection is selection.xenium_explorer_selection

    def test_root_import_defers_reader_until_access(self) -> None:
        code = """
import sys
import spatialdata_io
assert 'spatialdata_io.readers.xenium' not in sys.modules
names = ('xenium', 'xenium_aligned_image', 'xenium_explorer_selection')
assert all(name in spatialdata_io.__all__ for name in names)
from spatialdata_io import xenium, xenium_aligned_image, xenium_explorer_selection
from spatialdata_io.readers import xenium as package
assert xenium is package.xenium
assert xenium_aligned_image is package.xenium_aligned_image
assert xenium_explorer_selection is package.xenium_explorer_selection
"""
        result = subprocess.run(  # noqa: S603, RUF100 - Fixed test-owned code probes a fresh interpreter.
            [sys.executable, "-c", code], check=False, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr
