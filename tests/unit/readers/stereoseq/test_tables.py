"""Owned coordinate construction and final table linkage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.table import _linkage
from spatialdata_io.readers.stereoseq._tables import finalize_positions, link_table

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestFinalizePositions:
    def test_one_new_coordinate_buffer_and_unchanged_expression(self) -> None:
        native = np.array([[10, 20], [30, 40]], dtype=np.int64)
        table = ad.AnnData(csr_array([[11, 0], [0, 22]]))
        table.obsm["spatial"] = native
        table.layers["counts"] = csr_array([[5, 0], [0, 6]])
        table.raw = table.copy()
        expression, counts, raw = table.X, table.layers["counts"], table.raw

        finalize_positions(table, 500, "vendor_cell_center", calibration="Known 500 nm grid")

        assert table.X is expression
        assert table.layers["counts"] is counts
        assert table.raw is raw
        assert table.obsm["spatial_grid"] is native
        assert not np.shares_memory(table.obsm["spatial"], native)
        np.testing.assert_array_equal(native, [[10, 20], [30, 40]])
        np.testing.assert_array_equal(table.obsm["spatial"], [[5, 10], [15, 20]])
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(np.asarray(table.obsm["spatial"]), axis=0)), np.sqrt(200), rtol=1e-15
        )

    @pytest.mark.parametrize("destination", ["spatial_grid", POSITIONS_KEY])
    def test_collisions_fail_before_mutating_native(self, destination: str) -> None:
        native = np.array([[10, 20]], dtype=np.int64)
        table = ad.AnnData(csr_array([[1]]))
        table.obsm["spatial"] = native
        if destination == "spatial_grid":
            table.obsm[destination] = np.array([[9, 9]])
        else:
            table.uns[destination] = {"unrelated": "preserved"}

        with pytest.raises(ReaderFormatError, match="reserved"):
            finalize_positions(table, 500, "vendor_cell_center", calibration="Known 500 nm grid")
        assert table.obsm["spatial"] is native


class TestLinkTable:
    """Final linkage preserves the owned table and validates independent targets."""

    def test_exact_membership_and_single_linkage(self, mocker: MockerFixture) -> None:
        table = ad.AnnData(csr_array([[1, 0], [0, 2]]), obs=pd.DataFrame(index=["a", "b"]))
        target = pd.Index([200, 100], dtype=np.uint64)
        parse = mocker.spy(_linkage.TableModel, "parse")
        result = link_table(table, "cells_centroids", target)

        assert result is table
        assert result.obs["instance_id"].tolist() == [200, 100]
        assert result.obs["instance_id"].dtype == np.uint64
        assert result.obs["region"].cat.categories.tolist() == ["cells_centroids"]
        parse.assert_called_once()

    @pytest.mark.parametrize("column", ["region", "instance_id"])
    def test_conflicting_reserved_annotations(self, column: str) -> None:
        table = ad.AnnData(csr_array([[1]]), obs=pd.DataFrame({column: ["wrong"]}, index=["a"]))

        with pytest.raises(ReaderFormatError, match="reserved"):
            link_table(table, "cells_centroids", pd.Index([1]))

    def test_matching_reserved_annotations(self) -> None:
        table = ad.AnnData(csr_array([[1]]), obs=pd.DataFrame({"region": ["cells"], "instance_id": [1]}, index=["a"]))

        assert link_table(table, "cells", pd.Index([1])) is table
