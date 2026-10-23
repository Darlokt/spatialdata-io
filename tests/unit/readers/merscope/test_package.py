"""Intentional MERSCOPE package exports and lazy root import behavior."""

from __future__ import annotations

import importlib
import subprocess
import sys


class TestPackage:
    """Keep the reader public while storage and layout implementation stay private."""

    def test_only_reader_is_exported(self) -> None:
        package = importlib.import_module("spatialdata_io.readers.merscope")
        implementation = importlib.import_module("spatialdata_io.readers.merscope._reader")
        assert package.__all__ == ["merscope"]
        assert package.merscope is implementation.merscope

    def test_root_import_defers_reader_until_access(self) -> None:
        code = """
import sys
import spatialdata_io
assert 'spatialdata_io.readers.merscope' not in sys.modules
assert 'merscope' in spatialdata_io.__all__
from spatialdata_io import merscope
from spatialdata_io.readers.merscope import merscope as direct
assert merscope is direct
assert callable(merscope)
"""
        result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
