"""Intentional MCMICRO package exports and lazy root import behavior."""

from __future__ import annotations

import importlib
import subprocess
import sys


class TestPackage:
    """Keep the reader public while layout and storage implementation stay private."""

    def test_only_reader_is_exported(self) -> None:
        package = importlib.import_module("spatialdata_io.readers.mcmicro")
        implementation = importlib.import_module("spatialdata_io.readers.mcmicro._reader")
        assert package.__all__ == ["mcmicro"]
        assert package.mcmicro is implementation.mcmicro

    def test_root_import_defers_reader_until_access(self) -> None:
        code = """
import sys
import spatialdata_io
assert 'spatialdata_io.readers.mcmicro' not in sys.modules
assert 'mcmicro' in spatialdata_io.__all__
from spatialdata_io import mcmicro
from spatialdata_io.readers.mcmicro import mcmicro as direct
assert mcmicro is direct
assert callable(mcmicro)
"""
        result = subprocess.run(  # noqa: S603, RUF100 - Fixed test-owned code probes a fresh interpreter.
            [sys.executable, "-c", code], check=False, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr
