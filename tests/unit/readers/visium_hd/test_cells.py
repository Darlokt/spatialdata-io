"""Cell barcode and sparse nucleus arithmetic contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.sparse import csr_array

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium_hd._cells import _aggregate_counts, cell_ids, read_cell_table, read_nucleus_table
from tests.integration.readers.visium_hd._fixtures import write_counts

if TYPE_CHECKING:
    from pathlib import Path

    from anndata import AnnData
    from pytest_mock import MockerFixture


def _sources(root: Path, values: tuple[int, int] = (2, 3)) -> tuple[ArtifactFile, ArtifactFile, AnnData]:
    raw = root / "raw.h5"
    cell = root / "cell.h5"
    mapping = root / "mapping.parquet"
    write_counts(raw, np.array([[values[0], 0, 0], [values[1], 0, 0]], np.int32), ["bin0", "bin1"], "4.1.0")
    write_counts(cell, np.array([[1, 0, 0]], np.int32), ["cellid_000000001-1"], "4.1.0")
    pq.write_table(
        pa.table(
            {
                "square_002um": ["bin0", "bin1"],
                "cell_id": ["cellid_000000001-1"] * 2,
                "in_cell": [True, True],
                "in_nucleus": [True, True],
            }
        ),
        mapping,
    )
    return ArtifactFile(raw), ArtifactFile(mapping), read_cell_table(ArtifactFile(cell), make_var_names_unique=True)


class TestCellIds:
    """Accept only the documented fixed-width positive ID spelling."""

    def test_exact(self, tmp_path: Path) -> None:
        np.testing.assert_array_equal(
            cell_ids(pd.Index(["cellid_999999999-1", "cellid_000000001-1"]), ArtifactFile(tmp_path)), [999999999, 1]
        )

    @pytest.mark.parametrize(
        "barcode", ["cellid_0-1", "cellid_000000000-1", "cellid_000000001-2", "cellid_00000001a-1"]
    )
    def test_invalid(self, tmp_path: Path, barcode: str) -> None:
        with pytest.raises(ReaderFormatError, match="cell"):
            cell_ids(pd.Index([barcode]), ArtifactFile(tmp_path))


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
                source, np.array([], dtype=np.int64), np.array([], dtype=np.int64), 2, ArtifactFile(tmp_path / "raw.h5")
            )
        assert result.shape == (2, 1)
        assert result.nnz == 0
        assert result.dtype == np.dtype(np.float32)


class TestReadNucleusTable:
    """Compare sparse summation and failure behavior without densification."""

    def test_multiple_genes_are_canonical_without_densification(self, tmp_path: Path, mocker: MockerFixture) -> None:
        raw, mapping, population = _sources(tmp_path)
        write_counts(raw, np.array([[2, 4, 0], [3, 0, 5]], np.int32), ["bin0", "bin1"], "4.1.0")
        mocker.patch.object(csr_array, "toarray", side_effect=AssertionError("densification"))
        result = read_nucleus_table(raw, mapping, population, pd.Index([1]))
        assert isinstance(result.X, csr_array)
        assert result.X.has_canonical_format
        assert result.X.dtype == np.float32
        np.testing.assert_array_equal(result.X.data, [5, 4, 5])
        np.testing.assert_array_equal(result.X.indices, [0, 1, 2])
        np.testing.assert_array_equal(result.X.indptr, [0, 3])
        assert result.var_names.tolist() == ["Gene", "Gene-1", "Other"]

    def test_integer_sum_and_no_mutation(self, tmp_path: Path) -> None:
        raw, mapping, population = _sources(tmp_path)
        result = read_nucleus_table(raw, mapping, population, pd.Index([1]))
        assert isinstance(result.X, csr_array)
        assert result.X.dtype == np.float32
        assert isinstance(population.X, csr_array)
        assert result.X[0, 0] == 5
        assert population.X[0, 0] == 1
        assert result.var_names.tolist() == ["Gene", "Gene-1", "Other"]

    def test_no_nucleus_members_preserves_population_and_features(self, tmp_path: Path) -> None:
        raw, mapping, population = _sources(tmp_path)
        source_mapping = pq.read_table(mapping)
        pq.write_table(source_mapping.set_column(3, "in_nucleus", pa.array([False, False])), mapping)
        result = read_nucleus_table(raw, mapping, population, pd.Index([1]))
        assert isinstance(result.X, csr_array)
        assert result.X.shape == (1, 3)
        assert result.X.nnz == 0
        assert result.X.dtype == np.float32
        assert result.obs_names.tolist() == population.obs_names.tolist()
        assert result.var_names.tolist() == population.var_names.tolist()

    def test_float64_exact_limit_with_one_contributor(self, tmp_path: Path) -> None:
        raw, mapping, population = _sources(tmp_path)
        maximum = 2**53
        with h5py.File(raw, "r+") as source:
            del source["matrix/data"]
            source.create_dataset("matrix/data", data=np.array([maximum, 0], dtype=np.int64))
        source_mapping = pq.read_table(mapping)
        pq.write_table(source_mapping.set_column(3, "in_nucleus", pa.array([True, False])), mapping)
        result = read_nucleus_table(raw, mapping, population, pd.Index([1]))
        assert isinstance(result.X, csr_array)
        assert result.X[0, 0] == np.float32(maximum)

    @pytest.mark.parametrize("value", [2**62, 2**63])
    def test_float64_accumulation_precision_bound(self, tmp_path: Path, value: int) -> None:
        raw, mapping, population = _sources(tmp_path)
        with h5py.File(raw, "r+") as source:
            del source["matrix/data"]
            source.create_dataset("matrix/data", data=np.array([value, value], dtype=np.uint64))
        with pytest.raises(ReaderFormatError, match="float64 exact-sum bound"):
            read_nucleus_table(raw, mapping, population, pd.Index([1]))

    def test_float64_sum_preserves_unit_above_float32_precision(self, tmp_path: Path) -> None:
        raw, mapping, population = _sources(tmp_path, (2**24, 1))
        result = read_nucleus_table(raw, mapping, population, pd.Index([1]))
        assert isinstance(result.X, csr_array)
        assert result.X.dtype == np.dtype(np.float64)
        assert int(result.X[0, 0]) == 2**24 + 1

    def test_decoded_float32_input_is_not_reinterpreted_as_original_integer(self, tmp_path: Path) -> None:
        raw, mapping, population = _sources(tmp_path, (2**24 + 1, 0))
        result = read_nucleus_table(raw, mapping, population, pd.Index([1]))
        assert isinstance(result.X, csr_array)
        assert int(result.X[0, 0]) == 2**24

    @pytest.mark.parametrize("fault", ["duplicate", "unknown_bin", "unknown_cell", "missing"])
    def test_mapping_membership(self, tmp_path: Path, fault: str) -> None:
        raw, mapping, population = _sources(tmp_path)
        source = pq.read_table(mapping).to_pydict()
        if fault == "missing":
            source = {key: values[:1] for key, values in source.items()}
        elif fault == "unknown_cell":
            source["cell_id"][1] = "cellid_000000002-1"
        else:
            source["square_002um"][1] = "bin0" if fault == "duplicate" else "absent"
        pq.write_table(pa.table(source), mapping)
        with pytest.raises(ReaderFormatError, match=r"mapping|mapped|barcode"):
            read_nucleus_table(raw, mapping, population, pd.Index([1]))

    @pytest.mark.parametrize("field", ["id", "genome", "feature_type"])
    def test_feature_mismatch(self, tmp_path: Path, field: str) -> None:
        raw, mapping, population = _sources(tmp_path)
        with h5py.File(raw, "a") as handle:
            handle[f"matrix/features/{field}"][0] = b"wrong"
        with pytest.raises(ReaderFormatError, match="feature IDs"):
            read_nucleus_table(raw, mapping, population, pd.Index([1]))
