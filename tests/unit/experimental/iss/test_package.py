"""ISS intentional package exports."""

from __future__ import annotations

import importlib

from spatialdata_io.experimental.iss import iss


class TestPackage:
    """Keep only the public reader in the package export list."""

    def test_public_entry_point(self) -> None:
        package = importlib.import_module("spatialdata_io.experimental.iss")
        implementation = importlib.import_module("spatialdata_io.experimental.iss._reader")
        assert package.__all__ == ["iss"]
        assert iss is implementation.iss
