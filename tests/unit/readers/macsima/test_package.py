"""Intentional MACSima package exports and lazy root import behavior."""

from __future__ import annotations

import importlib
import subprocess
import sys


class TestPackage:
    """Keep the reader public while layout, metadata, and planning stay private."""

    def test_only_reader_is_exported(self) -> None:
        package = importlib.import_module("spatialdata_io.readers.macsima")
        implementation = importlib.import_module("spatialdata_io.readers.macsima._reader")
        assert package.__all__ == ["macsima"]
        assert package.macsima is implementation.macsima

    def test_root_import_defers_reader_until_access(self) -> None:
        code = """
import sys
import spatialdata_io
assert 'spatialdata_io.readers.macsima' not in sys.modules
assert 'macsima' in spatialdata_io.__all__
from spatialdata_io import macsima
from spatialdata_io.readers.macsima import macsima as direct
assert macsima is direct
assert callable(macsima)
"""
        result = subprocess.run(  # noqa: S603, RUF100 - Fixed test-owned code probes a fresh interpreter.
            [sys.executable, "-c", code], check=False, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr
