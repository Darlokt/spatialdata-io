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
from spatialdata_io.readers.seqfish._schemas import FLOAT_DTYPE
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

    @pytest.mark.parametrize("batch_rows", [1, 2, 10])
    def test_exact_mixed_count_columns_and_cross_batch_promotion(
        self, tmp_path: Path, mocker: MockerFixture, batch_rows: int
    ) -> None:
        # 16 bytes per feature and row; three features.
        mocker.patch("spatialdata_io.readers.seqfish._tables._BATCH_MEMORY_BYTES", 16 * 3 * batch_rows)
        counts = f",digits,decimal,exponent\n2,0,2.0,3e0\n1,{2**63},0.0,16777217e0\n"
        result = read_table(*_sources(tmp_path, counts))
        assert isinstance(result.X, sparse.csr_array)
        assert result.X.has_canonical_format
        assert result.X.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(result.X.data, np.array([2, 3, 2**63, 2**24 + 1], dtype=np.float64))
        np.testing.assert_array_equal(result.X.indices, [1, 2, 0, 2])
        np.testing.assert_array_equal(result.X.indptr, [0, 2, 4])
        assert result.var_names.tolist() == ["digits", "decimal", "exponent"]
        assert result.obs["instance_id"].tolist() == [2, 1]

    @pytest.mark.parametrize("ids", [(2**53, 2**53 + 1), (2**63 - 1, 2**63), (2**64 - 2, 2**64 - 1)])
    def test_reordered_exact_boundary_populations(self, tmp_path: Path, ids: tuple[int, int]) -> None:
        first, second = ids
        counts = f",A\n{first},11\n{second},13\n"
        cells = f"label,area,center_x,center_y\n{second},20,30,40\n{first},2,3,4\n"
        result = read_table(*_sources(tmp_path, counts, cells))
        assert result.obs["instance_id"].tolist() == [first, second]
        assert result.obs["area"].tolist() == [2, 20]
        np.testing.assert_array_equal(result.obsm["spatial"], [[3, 4], [30, 40]])

    @pytest.mark.parametrize(
        ("count_ids", "cell_ids"),
        [
            ((2**53, 2), (2**53 + 1, 2)),
            ((2**63 - 1, 2), (2**63, 2)),
            ((2**64 - 1, 2), (2**64 - 2, 2)),
            ((2**63, 2), (2**63 - 1, 2)),
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
        mocker.patch("spatialdata_io.readers.seqfish._tables._BATCH_MEMORY_BYTES", 1)
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
        assert result.obs["area"].dtype == FLOAT_DTYPE
        np.testing.assert_array_equal(result.obsm["spatial"], [[7, 5], [2, 3]])
        assert not result.uns

    def test_row_batches_are_bounded_only_by_bytes(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """20,001 single-feature rows fit one 8 MiB block; 1,000-row blocks give the identical table."""
        rows = 20_001
        counts = ",A\n" + "".join(f"{row + 1},{row % 3}\n" for row in range(rows))
        cells = "label,area,center_x,center_y\n" + "".join(f"{row + 1},1,{row},2\n" for row in range(rows))
        sources = _sources(tmp_path, counts, cells)
        blocks = mocker.spy(seqfish_tables, "_count_block")

        table = read_table(*sources)

        assert [call.args[0].num_rows for call in blocks.call_args_list] == [rows]
        # 16 bytes per feature and row; one feature.
        mocker.patch.object(seqfish_tables, "_BATCH_MEMORY_BYTES", 16 * 1000)
        blocks.reset_mock()
        small = read_table(*sources)
        assert len(blocks.call_args_list) == 21
        expected, actual = table.X, small.X
        assert isinstance(expected, sparse.csr_array)
        assert isinstance(actual, sparse.csr_array)
        assert expected.nnz == actual.nnz == 13_334
        assert expected.dtype == actual.dtype
        np.testing.assert_array_equal(expected.data, actual.data)
        np.testing.assert_array_equal(expected.indices, actual.indices)
        np.testing.assert_array_equal(expected.indptr, actual.indptr)

    def test_unsigned_mixed_batches(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch("spatialdata_io.readers.seqfish._tables._BATCH_MEMORY_BYTES", 1)
        large = 2**63
        counts = f",A\n65537,1\n{large},{large}\n"
        cells = f"label,area,center_x,center_y\n{large},1,2,3\n65537,4,5,6\n"
        result = read_table(*_sources(tmp_path, counts, cells))
        assert isinstance(result.X, sparse.csr_array)
        assert result.X.dtype == np.dtype(np.float32)
        assert result.X.data.tolist() == [1, large]
        assert result.obs["instance_id"].tolist() == [65537, large]

    @pytest.mark.parametrize("value", [2**53 + 1, 2**63 + 1, 2**64 - 1])
    def test_completed_counts_reject_inexact_float64(self, tmp_path: Path, value: int) -> None:
        with pytest.raises(ReaderFormatError, match="not exactly representable") as caught:
            read_table(*_sources(tmp_path, f",A\n1,{value}\n2,0\n"))
        assert caught.value.path == tmp_path / "counts.csv"
        assert caught.value.context.artifact == "CellxGene"

    @pytest.mark.parametrize(
        "counts",
        [
            ",A\n",
            ",A\n1,1\n1.0,2\n",
            ",A\n1,1.5\n",
            ",A\n1,-1\n",
            ",A\n1,1e-400\n",
            ",A\n1,18446744073709551616\n",
            ",A\n1,1,2\n",
            ",A,B\n1,1\n",
            ',A\n1,"bad\n',
            '"unterminated\n',
            ",A,A\n1,1,2\n",
            ",A\n1,0x10\n",
            ",A\n0x1,1\n",
            ",A\n1,1_0\n",
        ],
    )
    def test_invalid_counts(self, tmp_path: Path, counts: str) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            read_table(*_sources(tmp_path, counts))
        assert caught.value.context.artifact == "CellxGene"
        assert caught.value.path == tmp_path / "counts.csv"

    @pytest.mark.parametrize(
        "cells",
        [
            "label,area,center_x,center_y\n",
            "label,area,center_x,center_y\n1,2,3,4\n",
            "label,area,center_x,center_y\n1,2,3,4\n1,2,3,4\n",
            "label,area,center_x,center_y\n1,-1,3,4\n2,1,3,4\n",
            "label,area,center_x,center_y\n1,2,nan,4\n2,1,3,4\n",
            "label,area,center_x,center_y\n1,2,3\n",
            'label,area,center_x,center_y\n1,2,3,"bad\n',
        ],
    )
    def test_invalid_metadata(self, tmp_path: Path, cells: str) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            read_table(*_sources(tmp_path, cells=cells))
        assert caught.value.context.artifact == "CellCoordinates"
        assert caught.value.path == tmp_path / "cells.csv"

    def test_zero_area_and_quoted_records(self, tmp_path: Path) -> None:
        counts = ',"Gene,A"\n1,1e0\n'
        cells = 'label,area,center_x,center_y,extra\n1,0,0.10000000000000002,2,"line\nbreak"\n'
        result = read_table(*_sources(tmp_path, counts, cells))
        assert result.var_names.tolist() == ["Gene,A"]
        assert result.obs["area"].tolist() == [0]
        assert result.obsm["spatial"][0, 0] == 0.10000000000000002

    def test_quoted_multiline_records(self, tmp_path: Path) -> None:
        """Quoted fields may span lines; quoted numbers keep their exact spelling."""
        counts = ',"Gene,A"\r\n"1","2"\r\n'
        cells = 'label,area,center_x,center_y,ignored\r\n"1",0,"1.5",2,"line\nbreak"\r\n'

        result = read_table(*_sources(tmp_path, counts, cells))

        assert isinstance(result.X, sparse.csr_array)
        assert result.X.data.tolist() == [2]
        assert result.obs["instance_id"].tolist() == [1]
        assert result.var_names.tolist() == ["Gene,A"]
        np.testing.assert_array_equal(result.obsm["spatial"], [[1.5, 2]])

    @pytest.mark.parametrize(
        ("counts", "cells", "artifact", "found"),
        [
            (',A\n"1\n",2\n', _CELLS, "CellxGene", "'cell ID' value '1\\n'"),
            (',A\n1,"2\n"\n', _CELLS, "CellxGene", "'A' value '2\\n'"),
            (",A\n1,2\n", 'label,area,center_x,center_y\n" 1",0,1,2\n', "CellCoordinates", "'label' value ' 1'"),
        ],
    )
    def test_numbers_padded_with_whitespace_fail(
        self, tmp_path: Path, counts: str, cells: str, artifact: str, found: str
    ) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            read_table(*_sources(tmp_path, counts, cells))
        assert caught.value.context.artifact == artifact
        assert caught.value.found == found


class TestCountBlock:
    """Exercise exact column normalization at its bounded numerical owner."""

    @pytest.mark.parametrize("spelling", [str(2**64 - 1), f"{2**64 - 1}.0", f"{2**64 - 1}e0"])
    def test_full_uint64_fallback_and_ordinary_columns(self, tmp_path: Path, spelling: str) -> None:
        batch = pa.record_batch([["1"], ["9007199254740993.0"], [spelling], ["0"]], names=["", "A", "B", "C"])
        result = _count_block(batch, ("A", "B", "C"), ArtifactFile(tmp_path / "counts.csv"))
        assert result.dtype == np.dtype(np.uint64)
        np.testing.assert_array_equal(result.data, np.array([2**53 + 1, 2**64 - 1], dtype=np.uint64))
        np.testing.assert_array_equal(result.indices, [0, 1])
        np.testing.assert_array_equal(result.indptr, [0, 2])

    @pytest.mark.parametrize("tokens", [("0", "0.0", "0e0"), ("1", "2.0", "3e0")])
    def test_signed_zero_and_fallback_columns(self, tmp_path: Path, tokens: tuple[str, str, str]) -> None:
        batch = pa.record_batch(
            [["1", "2"], [tokens[0], "0"], [tokens[1], "0"], [tokens[2], "0"]], names=["", "A", "B", "C"]
        )
        result = _count_block(batch, ("A", "B", "C"), ArtifactFile(tmp_path / "counts.csv"))
        assert isinstance(result, sparse.csr_array)
        assert result.has_canonical_format
        assert result.shape == (2, 3)
        assert result.dtype == np.dtype(np.int64)
        if tokens[0] == "0":
            assert result.nnz == 0
            np.testing.assert_array_equal(result.indptr, [0, 0, 0])
        else:
            np.testing.assert_array_equal(result.data, [1, 2, 3])
            np.testing.assert_array_equal(result.indices, [0, 1, 2])
            np.testing.assert_array_equal(result.indptr, [0, 3, 3])


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
    def test_linkage_and_serialization(self, tmp_path: Path, region: str | None) -> None:
        table = read_table(*_sources(tmp_path))
        native = table.obsm["spatial"]
        expression = table.X
        result = finalize_table(table, frame="global", region=region, counts=ArtifactFile(tmp_path / "counts.csv"))
        assert result is table
        assert result.obsm["spatial"] is native
        assert result.X is expression
        assert result.uns["spatialdata_io_positions"]["coordinates"]["spatial"]["frame"] == "global"
        assert result.obs["instance_id"].tolist() == [2, 1]
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
        assert restored.obs["instance_id"].tolist() == [2, 1]
        assert restored.var_names.tolist() == ["B", "A"]
