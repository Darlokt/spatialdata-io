"""Cell barcode and sparse nucleus arithmetic contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium_hd._cells import _aggregate_counts, cell_ids

if TYPE_CHECKING:
    from pathlib import Path


class TestCellIds:
    """Accept only the documented fixed-width positive ID spelling."""

    def test_exact(self, tmp_path: Path) -> None:
        result = cell_ids(
            pd.Index(["cellid_999999999-1", "cellid_000000001-1"]), ArtifactFile(tmp_path), version="4.1.0"
        )
        np.testing.assert_array_equal(result, [999999999, 1])
        assert result.dtype == np.uint32

    def test_empty(self, tmp_path: Path) -> None:
        result = cell_ids(pd.Index([], dtype=str), ArtifactFile(tmp_path), version="4.1.0")
        assert result.shape == (0,)
        assert result.dtype == np.uint32

    @pytest.mark.parametrize(
        "barcode",
        [
            "cellid_0-1",
            "cellid_000000000-1",
            "cellid_000000001-2",
            "cellid_00000001a-1",
            "cellid_1000000000-1",
            "cellid_-00000001-1",
            "cellid_4294967297-1",
        ],
    )
    def test_invalid(self, tmp_path: Path, barcode: str) -> None:
        with pytest.raises(ReaderFormatError, match="cell"):
            cell_ids(pd.Index([barcode]), ArtifactFile(tmp_path), version="4.1.0")


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
