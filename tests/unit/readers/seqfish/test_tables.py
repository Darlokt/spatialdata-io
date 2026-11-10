"""Bounded seqFISH counts, exact independent alignment, and sparse output."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.seqfish import _tables as seqfish_tables
from spatialdata_io.readers.seqfish._tables import _count_block, finalize_table, read_table, select_cells

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


_CELLS = "center_y,label,extra,area,center_x\n3,1,ignored,4,2\n5,2,ignored,6,7\n"


def _sources(
    path: Path,
    counts: str = ",B,A\n2,0,4\n1,7,0\n",
    cells: str = _CELLS,
) -> tuple[ArtifactFile, ArtifactFile]:
    count_path = path / "counts.csv"
    cell_path = path / "cells.csv"
    count_path.write_text(counts, encoding="utf-8")
    cell_path.write_text(cells, encoding="utf-8")
    return ArtifactFile(count_path), ArtifactFile(cell_path)


class TestReadTable:
    """Preserve count row/feature order while aligning metadata by exact identity."""

    @pytest.mark.parametrize("batch_bytes", [40, 64, 1024])
    def test_exact_count_columns_and_cross_batch_promotion(
        self, tmp_path: Path, mocker: MockerFixture, batch_bytes: int
    ) -> None:
        mocker.patch.object(seqfish_tables, "_BATCH_BYTES", batch_bytes)
        counts = f",B,A,C\n2,0,2,3\n1,{2**53},0,16777217\n"
        result = read_table(*_sources(tmp_path, counts))
        assert isinstance(result.X, sparse.csr_array)
        assert result.X.has_canonical_format
        assert result.X.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(result.X.data, np.array([2, 3, 2**53, 2**24 + 1], dtype=np.float64))
        np.testing.assert_array_equal(result.X.indices, [1, 2, 0, 2])
        np.testing.assert_array_equal(result.X.indptr, [0, 2, 4])
        assert result.var_names.tolist() == ["B", "A", "C"]
        assert result.obs["instance_id"].tolist() == [2, 1]

    @pytest.mark.parametrize("ids", [(2**53, 2**53 + 1), (2**63 - 1, 2**63), (2**64 - 2, 2**64 - 1), (1, 2**64 - 1)])
    def test_reordered_exact_boundary_populations(self, tmp_path: Path, ids: tuple[int, int]) -> None:
        first, second = ids
        counts = f",A\n{first},11\n{second},13\n"
        cells = f"label,area,center_x,center_y\n{second},20,30,40\n{first},2,3,4\n"
        result = read_table(*_sources(tmp_path, counts, cells))
        assert result.obs["instance_id"].tolist() == [first, second]
        assert result.obs["instance_id"].dtype == pd.ArrowDtype(pa.uint64())
        assert result.obs["area"].tolist() == [2, 20]
        np.testing.assert_array_equal(result.obsm["spatial"], [[3, 4], [30, 40]])

    @pytest.mark.parametrize(
        ("count_ids", "cell_ids"),
        [
            ((2**53, 2), (2**53 + 1, 2)),
            ((2**63 - 1, 2), (2**63 - 2, 2)),
            ((2**64 - 1, 2), (2**64 - 2, 2)),
            ((1, 2), (1,)),
            ((1,), (1, 2)),
        ],
    )
    def test_missing_extra_and_adjacent_boundary_ids_are_distinct(
        self, tmp_path: Path, count_ids: tuple[int, ...], cell_ids: tuple[int, ...]
    ) -> None:
        counts = ",A\n" + "".join(f"{identifier},1\n" for identifier in count_ids)
        cells = "label,area,center_x,center_y\n" + "".join(f"{identifier},1,2,3\n" for identifier in cell_ids)
        with pytest.raises(ReaderFormatError, match="identical unique cell populations"):
            read_table(*_sources(tmp_path, counts, cells))

    def test_sparse_order_and_alignment(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch.object(seqfish_tables, "_BATCH_BYTES", 8)
        result = read_table(*_sources(tmp_path))
        assert isinstance(result.X, sparse.csr_array)
        assert not isinstance(result.X, sparse.spmatrix)
        assert result.X.has_canonical_format
        np.testing.assert_array_equal(result.X.data, [4, 7])
        np.testing.assert_array_equal(result.X.indices, [1, 0])
        assert result.obs_names.tolist() == ["0", "1"]
        assert result.var_names.tolist() == ["B", "A"]
        assert result.obs["instance_id"].tolist() == [2, 1]
        assert result.obs["area"].tolist() == [6, 4]
        assert result.obs["area"].dtype == pd.ArrowDtype(pa.float64())
        np.testing.assert_array_equal(result.obsm["spatial"], [[7, 5], [2, 3]])
        assert not result.uns

    def test_count_batches_are_bounded_by_source_bytes(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """20,001 single-feature rows fit one 1 MiB batch; 4 KiB batches give the identical table."""
        rows = 20_001
        counts = ",A\n" + "".join(f"{row + 1},{row % 3}\n" for row in range(rows))
        cells = "label,area,center_x,center_y\n" + "".join(f"{row + 1},1,{row},2\n" for row in range(rows))
        sources = _sources(tmp_path, counts, cells)
        blocks = mocker.spy(seqfish_tables, "_count_block")

        table = read_table(*sources)

        assert [call.args[0].num_rows for call in blocks.call_args_list] == [rows]
        mocker.patch.object(seqfish_tables, "_BATCH_BYTES", 4096)
        blocks.reset_mock()
        small = read_table(*sources)
        assert len(blocks.call_args_list) == len(counts.encode()) // 4096 + 1
        assert max(call.args[0].num_rows for call in blocks.call_args_list) < 4096 // 4
        expected, actual = table.X, small.X
        assert isinstance(expected, sparse.csr_array)
        assert isinstance(actual, sparse.csr_array)
        assert expected.nnz == actual.nnz == 13_334
        assert expected.dtype == actual.dtype
        np.testing.assert_array_equal(expected.data, actual.data)
        np.testing.assert_array_equal(expected.indices, actual.indices)
        np.testing.assert_array_equal(expected.indptr, actual.indptr)

    def test_records_must_fit_one_batch(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch.object(seqfish_tables, "_BATCH_BYTES", 1024)
        with pytest.raises(ReaderFormatError, match="straddles two block boundaries") as caught:
            read_table(*_sources(tmp_path, ",A\n1," + "0" * 2048 + "\n"))
        assert caught.value.context.artifact == "CellxGene"

    @pytest.mark.parametrize(("large", "dtype"), [(2**24, "float32"), (2**24 + 2, "float64"), (2**53, "float64")])
    def test_count_storage_boundaries_across_batches(
        self, tmp_path: Path, mocker: MockerFixture, large: int, dtype: str
    ) -> None:
        mocker.patch.object(seqfish_tables, "_BATCH_BYTES", 16)
        counts = f",A\n65537,1\n3,{large}\n"
        cells = "label,area,center_x,center_y\n3,1,2,3\n65537,4,5,6\n"
        result = read_table(*_sources(tmp_path, counts, cells))
        assert isinstance(result.X, sparse.csr_array)
        assert result.X.dtype == np.dtype(dtype)
        assert result.X.data.tolist() == [1, large]
        assert result.obs["instance_id"].tolist() == [65537, 3]

    @pytest.mark.parametrize("value", [2**53 + 1, 2**53 + 2, 2**63, 2**63 + 1, 2**64 - 1])
    def test_completed_counts_reject_above_float64_integer_range(self, tmp_path: Path, value: int) -> None:
        with pytest.raises(ReaderFormatError, match=r"counts above 2\*\*53") as caught:
            read_table(*_sources(tmp_path, f",A\n1,{value}\n2,0\n"))
        assert caught.value.path == tmp_path / "counts.csv"
        assert caught.value.context.artifact == "CellxGene"

    @pytest.mark.parametrize(
        ("counts", "found"),
        [
            (",A\n", "no count rows"),
            (",A\n1,1\n0,2\n", "nonpositive cell IDs"),
            (",A\n1,1\n1,2\n", "duplicate cell IDs"),
            (",A\n1,2.0\n", "invalid value '2.0'"),
            (",A\n1,1,2\n", "Expected 2 columns, got 3"),
        ],
    )
    def test_invalid_counts(self, tmp_path: Path, counts: str, found: str) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            read_table(*_sources(tmp_path, counts))
        assert found in caught.value.found
        assert caught.value.context.artifact == "CellxGene"
        assert caught.value.path == tmp_path / "counts.csv"

    @pytest.mark.parametrize(
        ("cells", "found"),
        [
            ("label,area,center_x,center_y\n", "different count and metadata IDs"),
            ("label,area,center_x,center_y\n1,2,3,4\n", "different count and metadata IDs"),
            ("label,area,center_x,center_y\n0,2,3,4\n", "nonpositive cell IDs"),
            ("label,area,center_x,center_y\n-1,2,3,4\n", "invalid value '-1'"),
            ("label,area,center_x,center_y\n1,2,3,4\n1,2,3,4\n", "duplicate cell IDs"),
            ("label,area,center_x,center_y\n1,-1,3,4\n2,1,3,4\n", "negative cell area"),
            ("label,area,center_x,center_y\n1,2,nan,4\n2,1,3,4\n", "nonfinite 'center_x' values"),
            ("label,area,center_x,center_y\n1,2,3,inf\n2,1,3,4\n", "nonfinite 'center_y' values"),
        ],
    )
    def test_invalid_metadata(self, tmp_path: Path, cells: str, found: str) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            read_table(*_sources(tmp_path, cells=cells))
        assert found in caught.value.found
        assert caught.value.context.artifact == "CellCoordinates"
        assert caught.value.path == tmp_path / "cells.csv"

    def test_quoted_fields_and_multiline_annotations(self, tmp_path: Path) -> None:
        """Quoted names and numbers parse like unquoted ones; an ignored annotation may span lines."""
        counts = ',"Gene,A",B\r\n"1","2",3\r\n'
        cells = '"",label,area,center_x,center_y,ignored\r\n0,"1",0,"1.5",2,"line\nbreak"\r\n'

        result = read_table(*_sources(tmp_path, counts, cells))

        assert isinstance(result.X, sparse.csr_array)
        assert result.X.data.tolist() == [2, 3]
        assert result.obs["instance_id"].tolist() == [1]
        assert result.obs["area"].tolist() == [0]
        assert result.var_names.tolist() == ["Gene,A", "B"]
        np.testing.assert_array_equal(result.obsm["spatial"], [[1.5, 2]])


class TestCountBlock:
    """Convert the typed count columns of one batch, never its cell IDs."""

    def test_counts_of_a_typed_batch(self) -> None:
        batch = pa.record_batch(
            [pa.array([7, 9]), pa.array([0, 2**64 - 1], pa.uint64()), pa.array([0, 0], pa.uint64())],
            names=["", "A", "B"],
        )
        result = _count_block(batch)
        assert isinstance(result, sparse.csr_array)
        assert result.has_canonical_format
        assert result.shape == (2, 2)
        assert result.dtype == np.dtype(np.uint64)
        np.testing.assert_array_equal(result.data, np.array([2**64 - 1], dtype=np.uint64))
        np.testing.assert_array_equal(result.indices, [0])
        np.testing.assert_array_equal(result.indptr, [0, 0, 1])


class TestSelectCells:
    """Keep selected rows with their names and reselect the exact count dtype."""

    @pytest.mark.parametrize(("kept", "dtype"), [([2], np.float32), ([1], np.float64), ([1, 2], np.float64)])
    def test_rows_names_and_count_dtype(self, tmp_path: Path, kept: list[int], dtype: type[np.floating]) -> None:
        table = read_table(*_sources(tmp_path, counts=f",B,A\n2,0,4\n1,{2**24 + 1},0\n"))
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.dtype == np.dtype(np.float64)

        result = select_cells(table, pd.Index(kept), counts=ArtifactFile(tmp_path / "counts.csv"))

        assert result is not table
        assert result.obs["instance_id"].tolist() == [i for i in (2, 1) if i in kept]
        assert result.obs_names.tolist() == [name for name, i in (("0", 2), ("1", 1)) if i in kept]
        assert isinstance(result.X, sparse.csr_array)
        assert result.X.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(result.X.data, [v for v, i in ((4, 2), (2**24 + 1, 1)) if i in kept])
        assert table.n_obs == 2


class TestFinalizeTable:
    """Normalize final owned annotations and link only to the selected target."""

    @pytest.mark.parametrize("region", [None, "Roi1_Segmentation"])
    @pytest.mark.parametrize("identifiers", [(2, 1), (2**64 - 1, 2**63)])
    def test_linkage_and_serialization(self, tmp_path: Path, region: str | None, identifiers: tuple[int, int]) -> None:
        first, second = identifiers
        counts = f",B,A\n{first},0,4\n{second},7,0\n"
        cells = f"label,area,center_x,center_y\n{second},4,2,3\n{first},6,7,5\n"
        table = read_table(*_sources(tmp_path, counts, cells))
        native = table.obsm["spatial"]
        expression = table.X
        result = finalize_table(table, frame="global", region=region, counts=ArtifactFile(tmp_path / "counts.csv"))
        assert result is table
        assert result.obsm["spatial"] is native
        assert result.X is expression
        assert result.uns["spatialdata_io_positions"]["coordinates"]["spatial"]["frame"] == "global"
        assert result.obs["instance_id"].tolist() == list(identifiers)
        assert result.obs["instance_id"].dtype == np.dtype(np.uint64)
        assert isinstance(result.X, sparse.csr_array)
        if region is None:
            assert "spatialdata_attrs" not in result.uns
            assert "region" not in result.obs
            assert result.obs.columns.tolist() == ["area", "instance_id"]
        else:
            assert result.uns["spatialdata_attrs"]["region"] == region
            assert result.obs["region"].tolist() == [region, region]
            assert result.obs.columns.tolist() == ["area", "region", "instance_id"]
        destination = tmp_path / "table.h5ad"
        result.write_h5ad(destination)
        restored = ad.read_h5ad(destination)
        assert restored.obs["instance_id"].tolist() == list(identifiers)
        assert restored.obs["instance_id"].dtype == np.dtype(np.uint64)
        assert restored.var_names.tolist() == ["B", "A"]
