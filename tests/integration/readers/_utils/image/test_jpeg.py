"""Integration tests for delayed JPEG decoding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import imagecodecs
import numpy as np
import pytest

from spatialdata_io.readers._utils.image import read_jpeg
from spatialdata_io.readers._utils.image._exceptions import RasterFormatError

if TYPE_CHECKING:
    from pathlib import Path


class TestReadJpeg:
    """Decode supported JPEG frames through one forced whole-frame task."""

    @pytest.mark.parametrize("channels", [1, 3])
    def test_matches_forced_decoder(self, tmp_path: Path, channels: int) -> None:
        path = tmp_path / f"channels-{channels}.jpg"
        shape = (9, 11) if channels == 1 else (9, 11, channels)
        source = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        path.write_bytes(imagecodecs.jpeg8_encode(source, level=90))
        expected = imagecodecs.jpeg8_decode(path.read_bytes())

        result = read_jpeg(path)

        assert result.data.numblocks == (1,) * result.data.ndim
        expected_axes = ("y", "x") if channels == 1 else ("y", "x", "c")
        assert result.axes == expected_axes
        np.testing.assert_array_equal(result.data.compute(), expected)

    def test_rejects_png_content(self, tmp_path: Path) -> None:
        path = tmp_path / "png.jpg"
        path.write_bytes(imagecodecs.png_encode(np.zeros((3, 4), dtype=np.uint8)))

        with pytest.raises(RasterFormatError, match="JPEG SOI"):
            read_jpeg(path)

    def test_accepts_jpeg_content_without_jpeg_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "image.data"
        source = np.arange(99, dtype=np.uint8).reshape(9, 11)
        path.write_bytes(imagecodecs.jpeg8_encode(source))

        result = read_jpeg(path)

        np.testing.assert_array_equal(
            result.data.compute(),
            imagecodecs.jpeg8_decode(path.read_bytes()),
        )
