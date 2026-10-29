"""Grid coverage and exact source-order sparse bin alignment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array, spmatrix

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium_hd._bins import (
    iter_position_batches,
    read_bin_table,
    read_grid_shape,
    read_positions,
)
from spatialdata_io.readers.visium_hd._layout import BinLayout
from spatialdata_io.readers.visium_hd._transforms import SlideFrame
from tests.integration.readers.visium_hd._fixtures import write_bundle, write_counts

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _positions(root: Path) -> ArtifactFile:
    return ArtifactFile(root / "binned_outputs/square_002um/spatial/tissue_positions.parquet")


class TestReadGridShape:
    """Parquet footer grid bounds and bounded fallback."""

    def test_footer_statistics(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        mocker.patch.object(pq.ParquetFile, "iter_batches", side_effect=AssertionError("unexpected positions decoding"))
        assert read_grid_shape(_positions(tmp_path)) == (17, 17)

    def test_projected_fallback(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        path = _positions(tmp_path)
        pq.write_table(pq.read_table(path), path, write_statistics=False)
        assert read_grid_shape(path) == (17, 17)

    @pytest.mark.parametrize("invalid", ["empty", "null"])
    def test_invalid(self, tmp_path: Path, invalid: str) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        path = _positions(tmp_path)
        table = pq.read_table(path)
        if invalid == "empty":
            table = table.slice(0, 0)
        else:
            table = table.set_column(
                2, pa.field("array_row", pa.uint32()), pa.array([None] * table.num_rows, type=pa.uint32())
            )
        pq.write_table(table, path, write_statistics=False)
        with pytest.raises(ReaderFormatError):
            read_grid_shape(path)

    def test_malformed_parquet(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "positions.parquet")
        path.write_bytes(b"not parquet")
        with pytest.raises(ReaderFormatError, match="positions Parquet"):
            read_grid_shape(path)


class TestIterPositionBatches:
    """Complete grid coverage, projection and resource ownership."""

    def test_projected_reordered(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        path = _positions(tmp_path)
        table = pq.read_table(path).take(np.arange(288, -1, -1)).append_column("extra", pa.array([True] * 289))
        pq.write_table(table, path)
        batches = list(iter_position_batches(path, (17, 17)))
        assert sum(batch.num_rows for batch in batches) == 289
        assert "extra" not in batches[0].schema.names

    @pytest.mark.parametrize("mode", ["missing", "row", "col", "duplicate", "nonfinite"])
    def test_invalid_grid(self, tmp_path: Path, mode: str) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        path = _positions(tmp_path)
        table = pq.read_table(path)
        if mode == "missing":
            table = table.slice(1)
        elif mode == "nonfinite":
            column = np.zeros(289)
            column[0] = np.nan
            table = table.set_column(4, table.schema.field(4), pa.array(column))
        elif mode == "duplicate":
            table = table.take(pa.array([1, *range(1, 289)]))
        else:
            name = "array_row" if mode == "row" else "array_col"
            column = table[name].to_numpy().copy()
            column[0] = 17
            table = table.set_column(table.schema.get_field_index(name), table.schema.field(name), pa.array(column))
        pq.write_table(table, path)
        with pytest.raises(ReaderFormatError):
            list(iter_position_batches(path, (17, 17)))

    def test_projected_batches_close_on_grid_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        path = _positions(tmp_path)
        table = pq.read_table(path)
        # Duplicate lies in a later decoded batch, exercising cross-batch coverage.
        indices = list(range(289))
        indices[40] = 0
        pq.write_table(table.take(pa.array(indices)), path)
        source = pq.ParquetFile(path)
        mocker.patch("spatialdata_io.readers.visium_hd._bins.POSITION_BATCH_SIZE", 37)
        mocker.patch("spatialdata_io.readers.visium_hd._bins.pq.ParquetFile", return_value=source)
        mocker.patch(
            "spatialdata_io.readers.visium_hd._bins.pq.read_table",
            side_effect=AssertionError("unbounded positions read"),
        )
        with pytest.raises(ReaderFormatError, match="duplicate grid"):
            list(iter_position_batches(path, (17, 17)))
        assert source.closed

    def test_malformed_parquet(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "positions.parquet")
        path.write_bytes(b"not parquet")
        with pytest.raises(ReaderFormatError, match="positions Parquet"):
            list(iter_position_batches(path, (17, 17)))


class TestReadPositions:
    """Exact barcode alignment in matrix observation order."""

    def test_exact_source_order_and_empty(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        path = _positions(tmp_path)
        names = pd.Index(["s_002um_00001_00003-1", "s_002um_00000_00000-1"])
        result = read_positions(path, (17, 17), names, 2)
        assert result.index.equals(names)
        assert_array_equal(result["array_row"], [1, 0])
        assert_array_equal(result["array_col"], [3, 0])
        assert str(result["pxl_col_in_fullres"].dtype) == "double[pyarrow]"
        assert read_positions(path, (17, 17), pd.Index([], dtype=str), 2).empty

    def test_selected_barcode_mismatch(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        path = _positions(tmp_path)
        table = pq.read_table(path)
        names = table["barcode"].to_pylist()
        names[0] = "wrong"
        pq.write_table(table.set_column(0, table.schema.field(0), pa.array(names)), path)
        with pytest.raises(ReaderFormatError, match="barcode differs"):
            read_positions(path, (17, 17), pd.Index(["s_002um_00000_00000-1"]), 2)


class TestReadBinTable:
    """Calibrated sparse bin matrices without densification."""

    @pytest.mark.parametrize("kind", ["raw", "filtered"])
    @pytest.mark.parametrize("unique", [False, True])
    def test_sparse_values_and_physical_centres(self, tmp_path: Path, kind: str, *, unique: bool) -> None:
        bundle = write_bundle(tmp_path, sizes=(2,), segmented=False)
        directory = tmp_path / "binned_outputs/square_002um"
        layout = BinLayout(
            2,
            ArtifactFile(directory / f"{kind}_feature_bc_matrix.h5"),
            _positions(tmp_path),
            ArtifactFile(directory / "spatial/scalefactors_json.json"),
        )
        frame = SlideFrame(np.eye(3), np.eye(3), np.eye(3), 0.0, 0.0)
        table = read_bin_table(layout, (17, 17), frame, make_var_names_unique=unique)
        assert isinstance(table.X, csr_array)
        assert not isinstance(table.X, spmatrix)
        expected = bundle.raw[2] if kind == "raw" else bundle.raw[2][bundle.filtered_rows[2]]
        assert_array_equal(table.X.toarray(), expected)
        assert table.var_names.tolist() == (["Gene", "Gene-1", "Other"] if unique else ["Gene", "Gene", "Other"])
        assert_array_equal(
            table.obsm["spatial"],
            np.column_stack((table.obs["array_col"].to_numpy() * 2 + 1, table.obs["array_row"].to_numpy() * 2 + 1)),
        )

    def test_expression_stays_sparse(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        directory = tmp_path / "binned_outputs/square_002um"
        layout = BinLayout(
            2,
            ArtifactFile(directory / "raw_feature_bc_matrix.h5"),
            _positions(tmp_path),
            ArtifactFile(directory / "spatial/scalefactors_json.json"),
        )
        mocker.patch.object(csr_array, "toarray", side_effect=AssertionError("expression densification"))
        frame = SlideFrame(np.eye(3), np.eye(3), np.eye(3), 0.0, 0.0)
        table = read_bin_table(layout, (17, 17), frame, make_var_names_unique=True)
        assert isinstance(table.X, csr_array)
        assert table.X.nnz == 6

    def test_empty_matrix_rejected(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        directory = tmp_path / "binned_outputs/square_002um"
        count_path = directory / "filtered_feature_bc_matrix.h5"
        write_counts(count_path, np.zeros((0, 3), dtype=np.int32), [], "4.1.0")
        layout = BinLayout(
            2,
            ArtifactFile(count_path),
            _positions(tmp_path),
            ArtifactFile(directory / "spatial/scalefactors_json.json"),
        )
        frame = SlideFrame(np.eye(3), np.eye(3), np.eye(3), 0.0, 0.0)
        with pytest.raises(ReaderFormatError, match="positive int64 dimensions"):
            read_bin_table(layout, (17, 17), frame, make_var_names_unique=True)
