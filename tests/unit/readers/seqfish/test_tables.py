"""Bounded seqFISH counts, exact independent alignment, and sparse output."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.seqfish._schemas import FLOAT_DTYPE
from spatialdata_io.readers.seqfish._tables import finalize_table, read_table

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _sources(
    path: Path,
    counts: str = ",B,A\n2,0,4\n1,7,0\n",
    cells: str = "center_y,label,extra,area,center_x\n3,1,ignored,4,2\n5,2,ignored,6,7\n",
) -> tuple[ArtifactFile, ArtifactFile]:
    count_path = path / "counts.csv"
    cell_path = path / "cells.csv"
    count_path.write_text(counts, encoding="utf-8")
    cell_path.write_text(cells, encoding="utf-8")
    return ArtifactFile(count_path), ArtifactFile(cell_path)


class TestReadTable:
    """Preserve count row/feature order while aligning metadata by exact identity."""

    def test_sparse_order_and_alignment(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch("spatialdata_io.readers.seqfish._tables._MAX_BATCH_ROWS", 1)
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

    def test_unsigned_mixed_batches(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch("spatialdata_io.readers.seqfish._tables._MAX_BATCH_ROWS", 1)
        large = 2**63 + 1
        counts = f",A\n65537,1\n{large},{large}\n"
        cells = f"label,area,center_x,center_y\n{large},1,2,3\n65537,4,5,6\n"
        result = read_table(*_sources(tmp_path, counts, cells))
        assert isinstance(result.X, sparse.csr_array)
        assert result.X.dtype == np.dtype(np.uint64)
        assert result.X.data.tolist() == [1, large]
        assert result.obs["instance_id"].tolist() == [65537, large]

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
        ],
    )
    def test_invalid_counts(self, tmp_path: Path, counts: str) -> None:
        with pytest.raises(ReaderFormatError):
            read_table(*_sources(tmp_path, counts))

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
        with pytest.raises(ReaderFormatError):
            read_table(*_sources(tmp_path, cells=cells))

    def test_zero_area_and_quoted_records(self, tmp_path: Path) -> None:
        counts = ',"Gene,A"\n1,1e0\n'
        cells = 'label,area,center_x,center_y,extra\n1,0,0.10000000000000002,2,"line\nbreak"\n'
        result = read_table(*_sources(tmp_path, counts, cells))
        assert result.var_names.tolist() == ["Gene,A"]
        assert result.obs["area"].tolist() == [0]
        assert result.obsm["spatial"][0, 0] == 0.10000000000000002

    def test_oversized_header(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch("spatialdata_io.readers.seqfish._tables._MAX_HEADER_CHARACTERS", 2)
        with pytest.raises(ReaderFormatError, match="oversized header"):
            read_table(*_sources(tmp_path))


class TestFinalizeTable:
    """Normalize final owned annotations and link only to the selected target."""

    @pytest.mark.parametrize("region", [None, "Roi1_Segmentation"])
    def test_linkage_and_serialization(self, tmp_path: Path, region: str | None) -> None:
        table = read_table(*_sources(tmp_path))
        result = finalize_table(table, region=region, context=ReaderErrorContext("seqfish", "table"))
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
