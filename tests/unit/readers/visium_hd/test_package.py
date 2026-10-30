"""Intentional reader exports and the lazy root import."""

from __future__ import annotations

import importlib
import subprocess
import sys


class TestPackage:
    """Only the stable entry point is exported; root imports remain deferred."""

    def test_intentional_exports(self) -> None:
        package = importlib.import_module("spatialdata_io.readers.visium_hd")
        reader = importlib.import_module("spatialdata_io.readers.visium_hd._reader")
        assert package.__all__ == ["visium_hd"]
        assert package.visium_hd is reader.visium_hd

    def test_lazy_root(self) -> None:
        script = """
import sys
import spatialdata_io
assert 'spatialdata_io.readers.visium_hd' not in sys.modules
assert 'visium_hd' in spatialdata_io.__all__
from spatialdata_io import visium_hd
from spatialdata_io.readers.visium_hd import visium_hd as packaged
assert visium_hd is packaged
"""
        result = subprocess.run(  # noqa: S603, RUF100 - Fixed test-owned interpreter and script.
            [sys.executable, "-c", script], check=False, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr
