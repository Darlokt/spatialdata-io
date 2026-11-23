"""Nucleus table construction from 10x count matrices and Parquet barcode mappings."""

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
from spatialdata_io.readers.visium_hd._cells import read_cell_table, read_nucleus_table
from tests.integration.readers.visium_hd._fixtures import VERSION, write_counts

if TYPE_CHECKING:
    from pathlib import Path

    from anndata import AnnData
    from pytest_mock import MockerFixture


def _sources(root: Path, values: tuple[int, int] = (2, 3)) -> tuple[ArtifactFile, ArtifactFile, ArtifactFile, AnnData]:
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
    return (
        ArtifactFile(raw),
        ArtifactFile(cell),
        ArtifactFile(mapping),
        read_cell_table(ArtifactFile(cell), make_var_names_unique=True, version=VERSION),
    )


class TestReadNucleusTable:
    """Compare sparse summation and failure behavior without densification."""

    def test_literal_barcodes_null_assignments_and_reordered_population(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        raw, cell, mapping, _ = _sources(tmp_path)
        names = ["007", "NA", "NULL", "bin3"]
        write_counts(raw, np.array([[2, 0, 0], [0, 3, 0], [0, 0, 5], [7, 0, 0]], np.int32), names, VERSION)
        selected = ["cellid_999999999-1", "cellid_000000001-1"]
        write_counts(cell, np.array([[1, 0, 0], [0, 1, 0]], np.int32), selected, VERSION)
        population = read_cell_table(cell, make_var_names_unique=True, version=VERSION)
        source = pa.table(
            {
                "square_002um": names,
                "cell_id": [selected[1], None, selected[0], "cellid_000000002-1"],
                "in_cell": [True, False, True, True],
                "in_nucleus": [True, False, True, True],
            }
        )
        pq.write_table(source.take(pa.array([3, 2, 0, 1])), mapping, row_group_size=1)
        mocker.patch("spatialdata_io.readers.visium_hd._cells._BATCH_SIZE", 2)

        result = read_nucleus_table(
            raw, mapping, population, pd.Index([1, 2, 999999999]), population_path=cell, version=VERSION
        )

        assert result.obs_names.tolist() == selected
        assert result.obs["instance_id"].tolist() == [999999999, 1]
        assert result.var_names.tolist() == ["Gene", "Gene-1", "Other"]
        assert isinstance(result.X, csr_array)
        assert result.X.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(result.X.data, [5, 2])
        np.testing.assert_array_equal(result.X.indices, [2, 0])
        np.testing.assert_array_equal(result.X.indptr, [0, 1, 2])

    def test_reordered_batch_cannot_repeat_a_barcode_in_an_aligned_batch(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        raw, cell, mapping, population = _sources(tmp_path)
        records = pq.read_table(mapping)
        pq.write_table(records.take(pa.array([1, 1])), mapping)
        mocker.patch("spatialdata_io.readers.visium_hd._cells._BATCH_SIZE", 1)
        with pytest.raises(ReaderFormatError, match="unknown or repeated"):
            read_nucleus_table(raw, mapping, population, pd.Index([1]), population_path=cell, version=VERSION)

    @pytest.mark.parametrize("fail", [False, True])
    def test_mapping_source_closes(self, tmp_path: Path, mocker: MockerFixture, *, fail: bool) -> None:
        raw, cell, mapping, population = _sources(tmp_path)
        source = pq.ParquetFile(mapping)
        mocker.patch("spatialdata_io.readers.visium_hd._cells.pq.ParquetFile", return_value=source)
        if fail:
            mocker.patch(
                "spatialdata_io.readers.visium_hd._cells.validate_mappings", side_effect=OSError("mapping read failed")
            )
            with pytest.raises(OSError, match="mapping read failed"):
                read_nucleus_table(raw, mapping, population, pd.Index([1]), population_path=cell, version=VERSION)
        else:
            read_nucleus_table(raw, mapping, population, pd.Index([1]), population_path=cell, version=VERSION)
        assert source.closed

    def test_multiple_genes_are_canonical_without_densification(self, tmp_path: Path, mocker: MockerFixture) -> None:
        raw, cell, mapping, population = _sources(tmp_path)
        write_counts(raw, np.array([[2, 4, 0], [3, 0, 5]], np.int32), ["bin0", "bin1"], "4.1.0")
        mocker.patch.object(csr_array, "toarray", side_effect=AssertionError("densification"))
        result = read_nucleus_table(raw, mapping, population, pd.Index([1]), population_path=cell, version=VERSION)
        assert isinstance(result.X, csr_array)
        assert result.X.has_canonical_format
        assert result.X.dtype == np.float32
        np.testing.assert_array_equal(result.X.data, [5, 4, 5])
        np.testing.assert_array_equal(result.X.indices, [0, 1, 2])
        np.testing.assert_array_equal(result.X.indptr, [0, 3])
        assert result.var_names.tolist() == ["Gene", "Gene-1", "Other"]

    def test_integer_sum_and_no_mutation(self, tmp_path: Path) -> None:
        raw, cell, mapping, population = _sources(tmp_path)
        result = read_nucleus_table(raw, mapping, population, pd.Index([1]), population_path=cell, version=VERSION)
        assert isinstance(result.X, csr_array)
        assert result.X.dtype == np.float32
        assert isinstance(population.X, csr_array)
        assert result.X[0, 0] == 5
        assert population.X[0, 0] == 1
        assert result.var_names.tolist() == ["Gene", "Gene-1", "Other"]

    def test_no_nucleus_members_preserves_population_and_features(self, tmp_path: Path) -> None:
        raw, cell, mapping, population = _sources(tmp_path)
        source_mapping = pq.read_table(mapping)
        pq.write_table(source_mapping.set_column(3, "in_nucleus", pa.array([False, False])), mapping)
        result = read_nucleus_table(raw, mapping, population, pd.Index([1]), population_path=cell, version=VERSION)
        assert isinstance(result.X, csr_array)
        assert result.X.shape == (1, 3)
        assert result.X.nnz == 0
        assert result.X.dtype == np.float32
        assert result.obs_names.tolist() == population.obs_names.tolist()
        assert result.var_names.tolist() == population.var_names.tolist()

    def test_float64_exact_limit_with_one_contributor(self, tmp_path: Path) -> None:
        raw, cell, mapping, population = _sources(tmp_path)
        maximum = 2**53
        with h5py.File(raw, "r+") as source:
            del source["matrix/data"]
            source.create_dataset("matrix/data", data=np.array([maximum, 0], dtype=np.int64))
        source_mapping = pq.read_table(mapping)
        pq.write_table(source_mapping.set_column(3, "in_nucleus", pa.array([True, False])), mapping)
        result = read_nucleus_table(raw, mapping, population, pd.Index([1]), population_path=cell, version=VERSION)
        assert isinstance(result.X, csr_array)
        assert result.X[0, 0] == np.float32(maximum)

    @pytest.mark.parametrize("value", [2**52 + 1, 2**53])
    def test_float64_accumulation_precision_bound(self, tmp_path: Path, value: int) -> None:
        raw, cell, mapping, population = _sources(tmp_path)
        with h5py.File(raw, "r+") as source:
            del source["matrix/data"]
            source.create_dataset("matrix/data", data=np.array([value, value], dtype=np.uint64))
        with pytest.raises(ReaderFormatError, match="float64 exact-sum bound"):
            read_nucleus_table(raw, mapping, population, pd.Index([1]), population_path=cell, version=VERSION)

    def test_float64_sum_preserves_unit_above_float32_precision(self, tmp_path: Path) -> None:
        raw, cell, mapping, population = _sources(tmp_path, (2**24, 1))
        result = read_nucleus_table(raw, mapping, population, pd.Index([1]), population_path=cell, version=VERSION)
        assert isinstance(result.X, csr_array)
        assert result.X.dtype == np.dtype(np.float64)
        assert int(result.X[0, 0]) == 2**24 + 1

    def test_hdf5_integers_above_float32_precision_remain_exact(self, tmp_path: Path) -> None:
        raw, cell, mapping, population = _sources(tmp_path, (2**24 + 1, 0))
        result = read_nucleus_table(raw, mapping, population, pd.Index([1]), population_path=cell, version=VERSION)
        assert isinstance(result.X, csr_array)
        assert result.X.dtype == np.float64
        assert int(result.X[0, 0]) == 2**24 + 1

    @pytest.mark.parametrize("fault", ["duplicate", "unknown_bin", "unknown_cell", "missing"])
    def test_mapping_membership(self, tmp_path: Path, fault: str) -> None:
        raw, cell, mapping, population = _sources(tmp_path)
        source = pq.read_table(mapping).to_pydict()
        if fault == "missing":
            source = {key: values[:1] for key, values in source.items()}
        elif fault == "unknown_cell":
            source["cell_id"][1] = "cellid_000000002-1"
        else:
            source["square_002um"][1] = "bin0" if fault == "duplicate" else "absent"
        pq.write_table(pa.table(source), mapping)
        with pytest.raises(ReaderFormatError, match=r"mapping|mapped|barcode"):
            read_nucleus_table(raw, mapping, population, pd.Index([1]), population_path=cell, version=VERSION)

    @pytest.mark.parametrize(
        ("assigned", "message"),
        [
            ("cellid_000000002-1", "unknown mapped cell ID"),
            ("cellid_00000000x-1", "noncanonical cell barcode"),
            ("cellid_000000000-1", "zero cell ID"),
        ],
    )
    def test_repeated_invalid_assignment_in_later_batch(
        self, tmp_path: Path, mocker: MockerFixture, assigned: str, message: str
    ) -> None:
        raw, cell, mapping, population = _sources(tmp_path)
        names = [f"bin{i}" for i in range(4)]
        write_counts(raw, np.ones((4, 3), np.int32), names, VERSION)
        pq.write_table(
            pa.table(
                {
                    "square_002um": names,
                    "cell_id": ["cellid_000000001-1"] * 2 + [assigned] * 2,
                    "in_cell": [True] * 4,
                    "in_nucleus": [True] * 4,
                }
            ),
            mapping,
            row_group_size=2,
        )
        mocker.patch("spatialdata_io.readers.visium_hd._cells._BATCH_SIZE", 2)
        with pytest.raises(ReaderFormatError, match=message) as error:
            read_nucleus_table(raw, mapping, population, pd.Index([1]), population_path=cell, version=VERSION)
        assert str(mapping) in str(error.value)
        assert VERSION in str(error.value)

    @pytest.mark.parametrize("field", ["id", "genome", "feature_type"])
    def test_feature_mismatch(self, tmp_path: Path, field: str) -> None:
        raw, cell, mapping, population = _sources(tmp_path)
        with h5py.File(raw, "a") as handle:
            handle[f"matrix/features/{field}"][0] = b"wrong"
        with pytest.raises(ReaderFormatError, match="feature IDs"):
            read_nucleus_table(raw, mapping, population, pd.Index([1]), population_path=cell, version=VERSION)
