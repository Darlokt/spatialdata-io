"""Tests for TIFF inspection boundary validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import RasterFormatError
from spatialdata_io.readers._utils.image import inspect_tiff

if TYPE_CHECKING:
    from pathlib import Path


class TestInspectTiff:
    """Reject invalid sources before opening tifffile metadata structures."""

    @pytest.mark.parametrize("signature", [b"", b"II", b"not tiff"])
    def test_rejects_invalid_magic_bytes(self, tmp_path: Path, signature: bytes) -> None:
        path = tmp_path / "image.tif"
        path.write_bytes(signature)

        with pytest.raises(RasterFormatError, match="signature"):
            inspect_tiff(path)
