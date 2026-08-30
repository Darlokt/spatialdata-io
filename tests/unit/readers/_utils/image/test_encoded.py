"""Tests for shared delayed whole-frame graph plumbing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spatialdata_io.readers._utils.image._encoded import encoded_result, validate_decoded_size
from spatialdata_io.readers._utils.image._exceptions import RasterFormatError
from spatialdata_io.readers._utils.image._types import _EncodedHeader


class TestValidateDecodedSize:
    """Reject invalid allocation geometry without imposing a pixel limit."""

    def test_accepts_positive_geometry(self) -> None:
        metadata = _EncodedHeader(
            shape=(5, 7, 4),
            dtype=np.dtype(np.uint16),
            axes=("y", "x", "c"),
        )

        validate_decoded_size(metadata)

    @pytest.mark.parametrize("shape", [(0, 4), (-1, 4)])
    def test_rejects_nonpositive_dimensions(self, shape: tuple[int, ...]) -> None:
        metadata = _EncodedHeader(
            shape=shape,
            dtype=np.dtype(np.uint8),
            axes=("y", "x"),
        )

        with pytest.raises(RasterFormatError, match="positive"):
            validate_decoded_size(metadata)

    def test_rejects_platform_allocation_overflow(self) -> None:
        metadata = _EncodedHeader(
            shape=(np.iinfo(np.intp).max, 2),
            dtype=np.dtype(np.uint8),
            axes=("y", "x"),
        )

        with pytest.raises(RasterFormatError, match="overflows"):
            validate_decoded_size(metadata)


class TestEncodedResult:
    """Construct one deterministic whole-frame source task."""

    def test_builds_one_source_chunk_without_opening_path(self) -> None:
        metadata = _EncodedHeader(
            shape=(5, 7),
            dtype=np.dtype(np.uint8),
            axes=("y", "x"),
        )

        result = encoded_result(Path("/not/opened/image.png"), "png", metadata)

        assert result.axes == metadata.axes
        assert result.data.chunks == ((5,), (7,))
        assert len(result.data.dask) == 1

    def test_rejects_unknown_internal_codec(self) -> None:
        metadata = _EncodedHeader(
            shape=(5, 7),
            dtype=np.dtype(np.uint8),
            axes=("y", "x"),
        )

        with pytest.raises(RasterFormatError, match="Unsupported"):
            encoded_result(
                Path("image.bin"),
                "gif",  # type: ignore[arg-type, bad-argument-type]
                metadata,
            )
