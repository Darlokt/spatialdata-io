"""Exact source CSC storage and final independent table linkage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium_hd import _tables as tables
from tests.integration.readers.visium_hd._fixtures import write_counts

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _source(tmp_path: Path) -> ArtifactFile:
    path = ArtifactFile(tmp_path / "matrix.h5")
    write_counts(path, np.array([[0, 1, 2], [3, 0, 4]], dtype=np.int32), ["first", "second"], "4.1.0")
    return path


def _replace(path: ArtifactFile, name: str, values: object) -> None:
    with h5py.File(path, "r+") as handle:
        del handle[f"matrix/{name}"]
        handle.create_dataset(f"matrix/{name}", data=values)


class TestReadCounts:
    """Prove source integers remain exact before Scanpy's float32 conversion."""

    def test_exact_values_dtype_and_batch_boundaries(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _source(tmp_path)
        mocker.patch.object(tables, "_COUNT_BATCH_SIZE", 1)
        result = tables.read_counts(path)
        assert isinstance(result.X, csr_array)
        assert result.X.dtype == np.float32
        assert_array_equal(csr_array(result.X).toarray(), [[0, 1, 2], [3, 0, 4]])
        assert_array_equal(result.obs_names, ["first", "second"])
        with h5py.File(path, "r+"):
            pass

    def test_exact_limit_and_zero_counts(self, tmp_path: Path) -> None:
        path = _source(tmp_path)
        _replace(path, "data", np.array([tables.MAX_EXACT_COUNT, 0, 0, 0], dtype=np.uint64))
        result = tables.read_counts(path)
        assert isinstance(result.X, csr_array)
        assert result.X.data.max() == tables.MAX_EXACT_COUNT
        for name, values in (
            ("data", np.array([], dtype=np.int32)),
            ("indices", np.array([], dtype=np.int32)),
            ("indptr", np.array([0, 0, 0], dtype=np.int64)),
        ):
            _replace(path, name, values)
        zero = tables.read_counts(path)
        assert isinstance(zero.X, csr_array)
        assert zero.X.nnz == 0

    @pytest.mark.parametrize("mode", ["duplicate", "unsorted", "duplicate-limit", "across-batch"])
    def test_noncanonical_coordinates(self, tmp_path: Path, mode: str, mocker: MockerFixture) -> None:
        path = _source(tmp_path)
        values = np.array(
            [tables.MAX_EXACT_COUNT, 1, 1, 1] if mode == "duplicate-limit" else [1, 1, 1, 1], dtype=np.int64
        )
        entries = np.array([2, 1, 0, 2] if mode == "unsorted" else [1, 1, 0, 2], dtype=np.int32)
        _replace(path, "data", values)
        _replace(path, "indices", entries)
        if mode == "across-batch":
            mocker.patch.object(tables, "_COUNT_BATCH_SIZE", 1)
        with pytest.raises(ReaderFormatError, match="duplicate or unsorted"):
            tables.read_counts(path)

    @pytest.mark.parametrize(
        "mode",
        [
            "missing",
            "group",
            "dtype",
            "ndim",
            "shape",
            "dimension",
            "length",
            "first",
            "last",
            "descending",
            "past-nnz",
            "negative-pointer",
            "feature-high",
            "feature-negative",
            "negative",
            "overflow",
            "barcodes",
            "io",
        ],
    )
    def test_malformed_source(self, tmp_path: Path, mode: str) -> None:
        path = _source(tmp_path)
        replacements: dict[str, tuple[str, object]] = {
            "dtype": ("data", np.array([1.0, 2.0, 3.0, 4.0])),
            "ndim": ("data", np.array([[1, 2, 3, 4]], dtype=np.int32)),
            "shape": ("shape", np.array([3, 2, 1])),
            "dimension": ("shape", np.array([0, 2])),
            "length": ("indices", np.array([1, 2, 0])),
            "first": ("indptr", np.array([1, 2, 4])),
            "last": ("indptr", np.array([0, 2, 3])),
            "descending": ("indptr", np.array([0, -1, 4])),
            "past-nnz": ("indptr", np.array([0, 5, 4])),
            "negative-pointer": ("indptr", np.array([0, -2, 4])),
            "feature-high": ("indices", np.array([1, 3, 0, 2])),
            "feature-negative": ("indices", np.array([-1, 2, 0, 2])),
            "negative": ("data", np.array([-1, 2, 3, 4])),
            "overflow": ("data", np.array([2**24 + 1, 2, 3, 4])),
            "barcodes": ("barcodes", np.array([b"same", b"same"])),
        }
        if mode in replacements:
            _replace(path, *replacements[mode])
        elif mode in {"missing", "group"}:
            with h5py.File(path, "r+") as handle:
                del handle["matrix/data"]
                if mode == "group":
                    handle.create_group("matrix/data")
        elif mode == "io":
            path.unlink()
        with pytest.raises((ReaderFormatError, FileNotFoundError)):
            tables.read_counts(path)


class TestLinkTable:
    """Link only against the independent materialized target IDs."""

    def test_linkage_and_serializable_categories(self, tmp_path: Path) -> None:
        result = tables.read_counts(_source(tmp_path))
        result.obs["instance_id"] = pd.Series([2, 5], index=result.obs_names, dtype="int64[pyarrow]")
        result = tables.link_table(result, "boundaries", pd.Index([2, 5]))
        assert result.uns["spatialdata_attrs"]["region"] == "boundaries"
        assert result.obs["region"].cat.categories.tolist() == ["boundaries"]
        result.write_h5ad(tmp_path / "roundtrip.h5ad")

    def test_membership(self, tmp_path: Path) -> None:
        result = tables.read_counts(_source(tmp_path))
        result.obs["instance_id"] = [2, 5]
        with pytest.raises(ReaderFormatError):
            tables.link_table(result, "boundaries", pd.Index([2, 3]))
