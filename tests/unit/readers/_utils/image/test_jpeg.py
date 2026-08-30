"""Tests for bounded JPEG metadata inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from spatialdata_io.readers._utils.image._encoded import MAX_HEADER_BYTES
from spatialdata_io.readers._utils.image._exceptions import RasterFormatError
from spatialdata_io.readers._utils.image._jpeg import inspect_jpeg_header

if TYPE_CHECKING:
    from pathlib import Path


def _jpeg_segment(marker: int, payload: bytes) -> bytes:
    return b"\xff" + bytes([marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def _jpeg_sof(*, precision: int = 8, components: int = 3, marker: int = 0xC0) -> bytes:
    component_data = b"".join(bytes([index + 1, 0x11, 0]) for index in range(components))
    payload = bytes([precision]) + (3).to_bytes(2, "big") + (5).to_bytes(2, "big")
    payload += bytes([components]) + component_data
    return b"\xff\xd8" + _jpeg_segment(marker, payload)


class TestInspectJpegHeader:
    """Validate supported SOF metadata and bounded marker traversal."""

    @pytest.mark.parametrize(
        ("marker", "components", "shape"),
        [
            (0xC0, 1, (3, 5)),
            (0xC1, 3, (3, 5, 3)),
            (0xC2, 3, (3, 5, 3)),
        ],
    )
    def test_predicts_decoder_shape(
        self,
        tmp_path: Path,
        marker: int,
        components: int,
        shape: tuple[int, ...],
    ) -> None:
        path = tmp_path / "image.jpg"
        path.write_bytes(
            b"\xff\xd8" + _jpeg_segment(0xE0, b"metadata") + _jpeg_sof(marker=marker, components=components)[2:]
        )

        metadata = inspect_jpeg_header(path)

        assert metadata.shape == shape
        assert metadata.dtype == np.dtype(np.uint8)

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            (b"not jpeg", "SOI"),
            (b"\xff\xd8\x00", "non-marker"),
            (b"\xff\xd8\xff", "Truncated"),
            (b"\xff\xd8\xff\x00", "stuffed zero"),
            (b"\xff\xd8\xff\xd9", "before a supported SOF"),
            (b"\xff\xd8\xff\xe0\x00\x01", "invalid segment length"),
            (b"\xff\xd8" + _jpeg_segment(0xC0, b"\x08"), "truncated SOF"),
            (_jpeg_sof(precision=12), "precision"),
            (_jpeg_sof(components=4), "component count"),
            (_jpeg_sof(marker=0xC3), "unsupported SOF"),
            (
                b"\xff\xd8" + _jpeg_segment(0xC0, b"\x08\x00\x03\x00\x05\x03\x01\x11\x00"),
                "inconsistent SOF",
            ),
        ],
    )
    def test_rejects_invalid_or_unsupported_frames(
        self,
        tmp_path: Path,
        payload: bytes,
        match: str,
    ) -> None:
        path = tmp_path / "image.jpg"
        path.write_bytes(payload)

        with pytest.raises(RasterFormatError, match=match):
            inspect_jpeg_header(path)

    def test_accepts_marker_fill_bytes_and_skips_standalone_markers(self, tmp_path: Path) -> None:
        path = tmp_path / "padded.jpg"
        path.write_bytes(b"\xff\xd8\xff\xd0\xff" + _jpeg_sof()[2:])

        assert inspect_jpeg_header(path).shape == (3, 5, 3)

    def test_enforces_header_scan_bound(self, tmp_path: Path) -> None:
        path = tmp_path / "oversized.jpg"
        segment = _jpeg_segment(0xE1, b"x" * (2**16 - 3))
        path.write_bytes(b"\xff\xd8" + segment * (MAX_HEADER_BYTES // len(segment) + 1))

        with pytest.raises(RasterFormatError, match="inspection limit"):
            inspect_jpeg_header(path)

    def test_bounds_marker_fill_scanning(self, tmp_path: Path) -> None:
        path = tmp_path / "marker-fill.jpg"
        path.write_bytes(b"\xff\xd8" + b"\xff" * MAX_HEADER_BYTES)

        with pytest.raises(RasterFormatError, match="inspection limit"):
            inspect_jpeg_header(path)
