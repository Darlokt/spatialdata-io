"""Integration tests for delayed PNG decoding."""

from __future__ import annotations

import struct
import zlib
from typing import TYPE_CHECKING

import imagecodecs
import numpy as np
import pytest

from spatialdata_io.readers._utils.errors import RasterFormatError
from spatialdata_io.readers._utils.image import read_png

if TYPE_CHECKING:
    from pathlib import Path


def _write_packed_png(path: Path, pixels: np.ndarray, *, bits: int, palette: bool, transparent: bool = False) -> None:
    """Encode packed PNG fixtures explicitly, including palette and transparency chunks."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        content = kind + payload
        return struct.pack(">I", len(payload)) + content + struct.pack(">I", zlib.crc32(content))

    height, width = pixels.shape
    header = struct.pack(">IIBBBBB", width, height, bits, 3 if palette else 0, 0, 0, 0)
    encoded = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
    if palette:
        colors = bytes(channel for i in range(2**bits) for channel in (i, 255 - i, (17 * i) % 256))
        encoded += chunk(b"PLTE", colors)
    if transparent:
        encoded += chunk(b"tRNS", bytes([255, 0] + [255] * (2**bits - 2)))
    # PNG stores each scanline's low-depth samples most-significant-bit first.
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint8)
    unpacked = ((pixels[..., None].astype(np.uint8) >> shifts) & 1).reshape(height, width * bits)
    packed = np.packbits(unpacked, axis=1)
    rows = b"".join(b"\0" + row.tobytes() for row in packed)
    path.write_bytes(encoded + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


class TestReadPng:
    """Decode supported PNG encodings exactly once and only on compute."""

    @pytest.mark.parametrize(
        ("channels", "dtype"),
        [
            *[(channels, np.dtype(np.uint8)) for channels in (1, 2, 3, 4)],
            *[(channels, np.dtype(np.uint16)) for channels in (1, 2, 3, 4)],
        ],
    )
    def test_matches_forced_decoder_for_direct_color_variants(
        self,
        tmp_path: Path,
        channels: int,
        dtype: np.dtype[np.generic],
    ) -> None:
        path = tmp_path / f"channels-{channels}-{dtype}.png"
        shape = (3, 4) if channels == 1 else (3, 4, channels)
        source = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        path.write_bytes(imagecodecs.png_encode(source))
        expected = imagecodecs.png_decode(path.read_bytes())

        result = read_png(path)

        assert result.data.numblocks == (1,) * result.data.ndim
        assert result.data.shape == expected.shape
        assert result.data.dtype == expected.dtype
        np.testing.assert_array_equal(result.data.compute(), expected)

    def test_preserves_uint16_grayscale(self, tmp_path: Path) -> None:
        path = tmp_path / "uint16.png"
        expected = np.arange(12, dtype=np.uint16).reshape(3, 4)
        path.write_bytes(imagecodecs.png_encode(expected))

        result = read_png(path)

        assert result.data.dtype == np.dtype(np.uint16)
        np.testing.assert_array_equal(result.data.compute(), expected)

    def test_expands_palette_and_transparency_exactly(self, tmp_path: Path) -> None:
        path = tmp_path / "palette.png"
        indices = np.arange(12, dtype=np.uint8).reshape(3, 4)
        _write_packed_png(path, indices, bits=8, palette=True, transparent=True)
        expected = imagecodecs.png_decode(path.read_bytes())

        result = read_png(path)

        assert result.axes == ("y", "x", "c")
        assert result.data.shape == (3, 4, 4)
        np.testing.assert_array_equal(result.data.compute(), expected)

    @pytest.mark.parametrize("bit_depth", [1, 2, 4])
    def test_expands_low_bit_depth_palette_exactly(self, tmp_path: Path, bit_depth: int) -> None:
        path = tmp_path / f"palette-{bit_depth}.png"
        indices = np.arange(12, dtype=np.uint8).reshape(3, 4) % (2**bit_depth)
        _write_packed_png(path, indices, bits=bit_depth, palette=True)
        expected = imagecodecs.png_decode(path.read_bytes())

        result = read_png(path)

        assert result.data.dtype == np.dtype(np.uint8)
        assert result.data.shape == (3, 4, 3)
        np.testing.assert_array_equal(result.data.compute(), expected)

    def test_decodes_one_bit_grayscale_to_uint8(self, tmp_path: Path) -> None:
        path = tmp_path / "grayscale-1.png"
        source = np.array([[False, True, False], [True, False, True]])
        _write_packed_png(path, source, bits=1, palette=False)

        result = read_png(path)

        assert result.data.dtype == np.dtype(np.uint8)
        np.testing.assert_array_equal(
            result.data.compute(),
            imagecodecs.png_decode(path.read_bytes()),
        )

    def test_rejects_jpeg_content(self, tmp_path: Path) -> None:
        path = tmp_path / "jpeg.png"
        path.write_bytes(imagecodecs.jpeg8_encode(np.zeros((3, 4), dtype=np.uint8)))

        with pytest.raises(RasterFormatError, match="PNG signature"):
            read_png(path)

    def test_accepts_png_content_without_png_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "image.data"
        expected = np.arange(12, dtype=np.uint8).reshape(3, 4)
        path.write_bytes(imagecodecs.png_encode(expected))

        result = read_png(path)

        np.testing.assert_array_equal(result.data.compute(), expected)
