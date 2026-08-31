"""Tests for bounded PNG metadata inspection."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import pytest

from spatialdata_io.readers._utils.errors import RasterFormatError
from spatialdata_io.readers._utils.image._encoded import MAX_HEADER_BYTES
from spatialdata_io.readers._utils.image._png import inspect_png_header

if TYPE_CHECKING:
    from pathlib import Path

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + b"\x00\x00\x00\x00"


def _write_png_header(
    path: Path,
    *,
    bit_depth: int = 8,
    color_type: int = 2,
    before_idat: bytes = b"",
) -> None:
    ihdr = struct.pack(">IIBBBBB", 5, 3, bit_depth, color_type, 0, 0, 0)
    path.write_bytes(_PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr) + before_idat + _png_chunk(b"IDAT", b""))


class TestInspectPngHeader:
    """Validate exact decoder metadata from bounded pre-IDAT bytes."""

    @pytest.mark.parametrize(
        ("bit_depth", "color_type", "extra", "shape"),
        [
            (1, 0, b"", (3, 5)),
            (16, 4, b"", (3, 5, 2)),
            (8, 2, b"", (3, 5, 3)),
            (16, 6, b"", (3, 5, 4)),
            (8, 3, _png_chunk(b"PLTE", b"\x00\x00\x00"), (3, 5, 3)),
            (
                8,
                3,
                _png_chunk(b"PLTE", b"\x00\x00\x00") + _png_chunk(b"tRNS", b"\x00"),
                (3, 5, 4),
            ),
        ],
    )
    def test_predicts_decoder_shape_and_dtype(
        self,
        tmp_path: Path,
        bit_depth: int,
        color_type: int,
        extra: bytes,
        shape: tuple[int, ...],
    ) -> None:
        path = tmp_path / "image.png"
        _write_png_header(path, bit_depth=bit_depth, color_type=color_type, before_idat=extra)

        metadata = inspect_png_header(path)

        assert metadata.shape == shape
        assert metadata.dtype == np.dtype(np.uint16 if bit_depth == 16 else np.uint8)

    @pytest.mark.parametrize(
        "payload",
        [
            b"not png",
            _PNG_SIGNATURE,
            _PNG_SIGNATURE + _png_chunk(b"IDAT", b""),
        ],
    )
    def test_rejects_missing_or_truncated_metadata(self, tmp_path: Path, payload: bytes) -> None:
        path = tmp_path / "image.png"
        path.write_bytes(payload)

        with pytest.raises(RasterFormatError):
            inspect_png_header(path)

    def test_rejects_invalid_palette_transparency(self, tmp_path: Path) -> None:
        path = tmp_path / "image.png"
        extra = _png_chunk(b"PLTE", b"\x00\x00\x00") + _png_chunk(b"tRNS", b"\x00\x01")
        _write_png_header(path, color_type=3, before_idat=extra)

        with pytest.raises(RasterFormatError, match="tRNS"):
            inspect_png_header(path)

    @pytest.mark.parametrize(
        ("bit_depth", "color_type", "extra", "match"),
        [
            (4, 2, b"", "unsupported color type"),
            (8, 3, b"", "no PLTE"),
            (8, 2, _png_chunk(b"PLTE", b""), "invalid PLTE"),
            (8, 0, _png_chunk(b"PLTE", b"\x00\x00\x00"), "invalid PLTE"),
            (1, 3, _png_chunk(b"PLTE", b"\x00" * 9), "invalid PLTE"),
            (8, 2, _png_chunk(b"IHDR", b"x" * 13), "more than one IHDR"),
            (8, 2, _png_chunk(b"ABCD", b""), "unsupported critical chunk"),
            (8, 2, _png_chunk(b"1abc", b""), "invalid chunk type"),
            (8, 2, _png_chunk(b"abca", b""), "lowercase reserved"),
            (
                8,
                3,
                _png_chunk(b"PLTE", b"\x00\x00\x00") + _png_chunk(b"tRNS", b"\x00") + _png_chunk(b"tRNS", b"\x00"),
                "more than one tRNS",
            ),
        ],
    )
    def test_rejects_unsupported_or_duplicate_chunks(
        self,
        tmp_path: Path,
        bit_depth: int,
        color_type: int,
        extra: bytes,
        match: str,
    ) -> None:
        path = tmp_path / "invalid.png"
        _write_png_header(path, bit_depth=bit_depth, color_type=color_type, before_idat=extra)

        with pytest.raises(RasterFormatError, match=match):
            inspect_png_header(path)

    @pytest.mark.parametrize(
        ("ihdr", "match"),
        [
            (b"short", "IHDR chunk length"),
            (struct.pack(">IIBBBBB", 5, 3, 8, 2, 1, 0, 0), "unsupported IHDR methods"),
        ],
    )
    def test_rejects_invalid_ihdr(self, tmp_path: Path, ihdr: bytes, match: str) -> None:
        path = tmp_path / "invalid-ihdr.png"
        path.write_bytes(_PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", b""))

        with pytest.raises(RasterFormatError, match=match):
            inspect_png_header(path)

    def test_rejects_iend_before_image_data(self, tmp_path: Path) -> None:
        path = tmp_path / "ended.png"
        ihdr = struct.pack(">IIBBBBB", 5, 3, 8, 2, 0, 0, 0)
        path.write_bytes(_PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IEND", b""))

        with pytest.raises(RasterFormatError, match="ended before"):
            inspect_png_header(path)

    @pytest.mark.parametrize(("color_type", "length", "shape"), [(0, 2, (3, 5, 2)), (2, 6, (3, 5, 4))])
    def test_predicts_alpha_channel_from_valid_transparency(
        self,
        tmp_path: Path,
        color_type: int,
        length: int,
        shape: tuple[int, ...],
    ) -> None:
        path = tmp_path / "transparent.png"
        _write_png_header(
            path,
            color_type=color_type,
            before_idat=_png_chunk(b"tRNS", bytes(length)),
        )

        assert inspect_png_header(path).shape == shape

    def test_enforces_header_scan_bound_before_allocation(self, tmp_path: Path) -> None:
        path = tmp_path / "oversized.png"
        oversized_text = _png_chunk(b"tEXt", b"x" * MAX_HEADER_BYTES)
        _write_png_header(path, before_idat=oversized_text)

        with pytest.raises(RasterFormatError, match="inspection limit"):
            inspect_png_header(path)
