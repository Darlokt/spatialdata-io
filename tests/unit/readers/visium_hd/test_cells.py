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
from spatialdata_io.readers.visium_hd._cells import cell_ids, read_cell_table, read_nucleus_table
from tests.integration.readers.visium_hd._fixtures import write_counts

if TYPE_CHECKING:
    from pathlib import Path

    from anndata import AnnData


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


class TestReadNucleusTable:
    """Compare sparse summation and failure behavior without densification."""

    def test_integer_sum_and_no_mutation(self, tmp_path: Path) -> None:
        raw, mapping, population = _sources(tmp_path)
        result = read_nucleus_table(raw, mapping, population, pd.Index([1]))
        assert isinstance(result.X, csr_array)
        assert result.X.dtype == np.float32
        assert isinstance(population.X, csr_array)
        assert result.X[0, 0] == 5
        assert population.X[0, 0] == 1
        assert result.var_names.tolist() == ["Gene", "Gene-1", "Other"]

    def test_int64_limit_with_one_contributor(self, tmp_path: Path) -> None:
        raw, mapping, population = _sources(tmp_path)
        maximum = np.iinfo(np.int64).max
        with h5py.File(raw, "r+") as source:
            del source["matrix/data"]
            source.create_dataset("matrix/data", data=np.array([maximum, 0], dtype=np.int64))
        source_mapping = pq.read_table(mapping)
        pq.write_table(source_mapping.set_column(3, "in_nucleus", pa.array([True, False])), mapping)
        result = read_nucleus_table(raw, mapping, population, pd.Index([1]))
        assert isinstance(result.X, csr_array)
        assert result.X[0, 0] == np.float32(maximum)

    @pytest.mark.parametrize("value", [2**62, 2**63])
    def test_int64_accumulation_overflow(self, tmp_path: Path, value: int) -> None:
        raw, mapping, population = _sources(tmp_path)
        with h5py.File(raw, "r+") as source:
            del source["matrix/data"]
            source.create_dataset("matrix/data", data=np.array([value, value], dtype=np.uint64))
        with pytest.raises(ReaderFormatError, match="int64"):
            read_nucleus_table(raw, mapping, population, pd.Index([1]))

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
