"""Sparse nucleus arithmetic contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.sparse import csr_array

from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium_hd._cells import _aggregate_counts

if TYPE_CHECKING:
    from pathlib import Path


class TestAggregateCounts:
    @pytest.mark.parametrize("source_dtype", ["int64", "float32", "float64"])
    @pytest.mark.parametrize(
        ("values", "expected_dtype"),
        [
            ((2, 3), "float32"),
            ((2**23, 2**23), "float32"),
            ((2**23 + 1, 2**23), "float64"),
            ((2**24, 1), "float64"),
            ((2**52, 2**52), "float64"),
        ],
    )
    def test_exact_arithmetic_and_source_ownership(
        self, tmp_path: Path, source_dtype: str, values: tuple[int, int], expected_dtype: str
    ) -> None:
        source = csr_array(np.array([[values[0], 2, 0], [values[1], 0, 3]], dtype=source_dtype))
        original = source.data.copy(), source.indices.copy(), source.indptr.copy()
        result = _aggregate_counts(
            source,
            np.array([0, 0], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            1,
            ArtifactFile(tmp_path / "raw.h5"),
            version="4.1.0",
        )
        assert result.dtype == np.dtype(expected_dtype)
        assert result.has_canonical_format
        assert int(result[0, 0]) == sum(int(value) for value in source[:, [0]].data)
        np.testing.assert_array_equal(result.indices, [0, 1, 2])
        np.testing.assert_array_equal(result.data[1:], [2, 3])
        for actual, expected in zip((source.data, source.indices, source.indptr), original, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_no_contributors_does_not_convert_unused_large_counts(self, tmp_path: Path) -> None:
        source = csr_array([[1e300]], dtype=np.float64)
        with np.errstate(over="raise", invalid="raise"):
            result = _aggregate_counts(
                source,
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                2,
                ArtifactFile(tmp_path / "raw.h5"),
                version="4.1.0",
            )
        assert result.shape == (2, 1)
        assert result.nnz == 0
        assert result.dtype == np.dtype(np.float32)
