"""Tests for shared delayed whole-frame graph plumbing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spatialdata_io.readers._utils.errors import RasterFormatError
from spatialdata_io.readers._utils.image._encoded import encoded_result, validate_decoded_size
from spatialdata_io.readers._utils.image._types import _EncodedHeader
from spatialdata_io.readers._utils.paths import ArtifactFile


class TestValidateDecodedSize:
    """Reject invalid allocation geometry without imposing a pixel limit."""

    def test_accepts_positive_geometry(self) -> None:
        metadata = _EncodedHeader(
            shape=(5, 7, 4),
            dtype=np.dtype(np.uint16),
            axes=("y", "x", "s"),
            color="rgba",
        )

        validate_decoded_size(metadata)

    @pytest.mark.parametrize("shape", [(0, 4), (-1, 4)])
    def test_rejects_nonpositive_dimensions(self, shape: tuple[int, ...]) -> None:
        metadata = _EncodedHeader(
            shape=shape,
            dtype=np.dtype(np.uint8),
            axes=("y", "x"),
            color=None,
        )

        with pytest.raises(RasterFormatError, match="positive"):
            validate_decoded_size(metadata)

    def test_rejects_platform_allocation_overflow(self) -> None:
        metadata = _EncodedHeader(
            shape=(np.iinfo(np.intp).max, 2),
            dtype=np.dtype(np.uint8),
            axes=("y", "x"),
            color=None,
        )

        with pytest.raises(RasterFormatError, match="overflows"):
            validate_decoded_size(metadata)


class TestEncodedResult:
    """Construct one deterministic whole-frame source task."""

    def test_builds_one_source_chunk_without_opening_path(self) -> None:
        metadata = _EncodedHeader(
            shape=(5, 7, 3),
            dtype=np.dtype(np.uint8),
            axes=("y", "x", "s"),
            color="rgb",
        )

        result = encoded_result(ArtifactFile(Path("/not/opened/image.png")), "png", metadata)

        assert result.axes == ("y", "x", "s")
        assert result.color == "rgb"
        assert result.data.chunks == ((5,), (7,), (3,))
        assert len(result.data.dask) == 1
