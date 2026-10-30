"""Tests for TIFF inspection boundary validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from spatialdata_io.readers._utils.errors import RasterFormatError
from spatialdata_io.readers._utils.image import inspect_tiff
from spatialdata_io.readers._utils.image.tiff._metadata import _level_metadata

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestInspectTiff:
    """Reject invalid sources before opening tifffile metadata structures."""

    @pytest.mark.parametrize("signature", [b"", b"II", b"not tiff"])
    def test_rejects_invalid_magic_bytes(self, tmp_path: Path, signature: bytes) -> None:
        path = tmp_path / "image.tif"
        path.write_bytes(signature)

        with pytest.raises(RasterFormatError, match="signature"):
            inspect_tiff(path)


class TestLevelMetadata:
    """Keep semantic dimensions stable when tifffile retains source singleton axes."""

    def test_unsqueezed_series_matches_squeezed_store(self, mocker: MockerFixture) -> None:
        level = mocker.Mock(spec=tifffile.TiffPageSeries)
        level.shape = (1, 1, 8, 1)
        level.axes = "TCYX"
        level.dtype = np.dtype(np.uint16)
        tiff = mocker.Mock(spec=tifffile.TiffFile)
        mocker.patch("spatialdata_io.readers._utils.image.tiff._metadata._native_chunks", return_value=(8, 1))
        metadata = _level_metadata(tiff, level, series_index=0, level_index=0)
        assert metadata.shape == (8, 1)
        assert metadata.axes == ("Y", "X")
        assert metadata.native_chunks == (8, 1)
