"""Representability and buffer ownership of regionprops centroid conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spatialdata_io.readers._utils.centroids import scale_regionprops_centroids
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile

_PATH = ArtifactFile(Path("measurements"))
_CONTEXT = ReaderErrorContext("test", "measurements")


class TestScaleRegionpropsCentroids:
    def test_anisotropic_and_output_view(self) -> None:
        native = np.array([[0, 1], [2, 3]], dtype=np.uint16)
        buffer = np.zeros((4, 2))
        out = buffer[::2]
        out[:] = native
        scale_regionprops_centroids(out, (2, 4), path=_PATH, context=_CONTEXT, frame="sample")
        np.testing.assert_array_equal(buffer, [[1, 6], [0, 0], [5, 14], [0, 0]])
        np.testing.assert_array_equal(native, [[0, 1], [2, 3]])

    def test_zero_and_signed_subnormal(self) -> None:
        native = np.array([[-0.5, -1.5], [0.5, 0.5]])
        result = native.copy()
        scale_regionprops_centroids(result, (5e-324, 5e-324), path=_PATH, context=_CONTEXT, frame="sample")
        np.testing.assert_array_equal(result, [[0, -5e-324], [5e-324, 5e-324]])
        assert not np.shares_memory(native, result)

    @pytest.mark.parametrize(("x", "spacing", "match"), [(2.0, 1e308, "nonfinite"), (0.0, 5e-324, "nonzero")])
    def test_unrepresentable(self, x: float, spacing: float, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match) as caught:
            scale_regionprops_centroids(np.array([[x, 0]]), (spacing, 1), path=_PATH, context=_CONTEXT, frame="sample")
        assert caught.value.path == _PATH
        assert caught.value.context == _CONTEXT
