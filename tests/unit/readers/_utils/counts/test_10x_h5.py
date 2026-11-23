"""Unit tests for HDF5 sparse count interpretation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import sparse

from spatialdata_io.readers._utils.counts._10x_h5 import _h5_counts
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from pathlib import Path


_CONTEXT = ReaderErrorContext(reader="test_reader", artifact="count matrix", detected_version="3")


def _compressed(values: list[float], indices: list[int], dtype: str) -> sparse.csr_array:
    buffers = (np.array(values, dtype=dtype), np.array(indices, dtype=np.int32), np.array([0, len(values)]))
    shape = (1, max(indices) + 1)
    return sparse.csr_array(buffers, shape=shape)


class TestH5Counts:
    @pytest.mark.parametrize("dtype", ["int32", "float32", "float64"])
    def test_rejects_repeated_entries_with_source_context(self, tmp_path: Path, dtype: str) -> None:
        path = tmp_path / "counts.h5"
        source = _compressed([2, 3, 4], [1, 0, 1], dtype)

        with pytest.raises(ReaderFormatError, match=r"repeated \(barcode, feature\) entries") as caught:
            _h5_counts(source, path, _CONTEXT)

        assert caught.value.context == _CONTEXT
        assert caught.value.path == path

    @pytest.mark.parametrize("values", [(0.5, 0.5), (-1.0, 2.0)])
    def test_repeated_entries_cannot_become_valid_counts(self, tmp_path: Path, values: tuple[float, float]) -> None:
        source = _compressed(list(values), [0, 0], "float64")
        with pytest.raises(ReaderFormatError, match="repeated"):
            _h5_counts(source, tmp_path / "counts.h5", _CONTEXT)

    def test_sorts_unique_entries_without_summing(self, tmp_path: Path) -> None:
        source = _compressed([2**30, 1], [1, 0], "float32")

        result = _h5_counts(source, tmp_path / "counts.h5", _CONTEXT)

        assert isinstance(result, sparse.csr_array)
        assert result.has_canonical_format
        assert result.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(result.data, [1, 2**30])

    @pytest.mark.parametrize(
        ("dtype", "value", "expected"),
        [
            ("bool", 1, "float32"),
            ("int32", 2**24, "float32"),
            ("int64", 2**24 + 1, "float64"),
            ("float32", 2**30, "float32"),
            ("float64", 7, "float32"),
            ("float64", 2**24 + 1, "float64"),
        ],
    )
    def test_stores_counts_as_float32_or_float64(self, tmp_path: Path, dtype: str, value: int, expected: str) -> None:
        result = _h5_counts(sparse.csr_array(np.array([[value]], dtype=dtype)), tmp_path / "c.h5", _CONTEXT)
        assert isinstance(result, sparse.csr_array)
        assert result.dtype == np.dtype(expected)
        assert int(result.data[0]) == value

    def test_canonical_float32_counts_are_reused(self, tmp_path: Path) -> None:
        data = sparse.csr_array(np.asarray([[0, 1, 100_000]], dtype=np.float32))
        assert _h5_counts(data, tmp_path / "counts.h5", _CONTEXT) is data

    @pytest.mark.parametrize("value", [-1, 0.5])
    def test_rejects_malformed_counts(self, tmp_path: Path, value: float) -> None:
        source = sparse.csr_array(np.asarray([[value]], dtype=np.float32))
        with pytest.raises(ReaderFormatError, match="negative or fractional"):
            _h5_counts(source, tmp_path / "counts.h5", _CONTEXT)

    def test_empty_counts(self, tmp_path: Path) -> None:
        result = _h5_counts(sparse.csr_array((2, 2), dtype=np.float32), tmp_path / "c.h5", _CONTEXT)
        assert isinstance(result, sparse.csr_array)
        assert result.shape == (2, 2)
