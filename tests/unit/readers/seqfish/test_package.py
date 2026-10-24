"""Intentional seqFISH package exports and lazy root import behavior."""

from __future__ import annotations

import importlib
import subprocess
import sys


class TestPackage:
    """Keep the reader public while storage and layout implementation stay private."""

    def test_only_reader_is_exported(self) -> None:
        package = importlib.import_module("spatialdata_io.readers.seqfish")
        implementation = importlib.import_module("spatialdata_io.readers.seqfish._reader")
        assert package.__all__ == ["seqfish"]
        assert package.seqfish is implementation.seqfish

    def test_root_import_defers_reader_until_access(self) -> None:
        code = """
import sys
import spatialdata_io
assert 'spatialdata_io.readers.seqfish' not in sys.modules
assert 'seqfish' in spatialdata_io.__all__
from spatialdata_io import seqfish
from spatialdata_io.readers.seqfish import seqfish as direct
assert seqfish is direct
assert callable(seqfish)
"""
        result = subprocess.run(  # noqa: S603, RUF100 - Fixed test-owned code probes a fresh interpreter.
            [sys.executable, "-c", code], check=False, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr
