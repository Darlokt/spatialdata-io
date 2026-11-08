"""Dropped cells, owned coordinate construction and final table linkage."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.table import _linkage
from spatialdata_io.readers.stereoseq._tables import finalize_positions, link_table, select_cells

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

CONTEXT = ReaderErrorContext("stereoseq", "cell counts")
PATH = Path("sample.cellbin.gef")


class TestSelectCells:
    """Rows of dropped contours leave the table; counts are re-finalized."""

    def test_all_retained_returns_the_same_table(self) -> None:
        table = ad.AnnData(csr_array(np.array([[1, 0], [0, 2]], dtype=np.float32)), obs=pd.DataFrame(index=["7", "3"]))

        assert select_cells(table, pd.Index([7, 3], dtype=np.uint64), context=CONTEXT, path=PATH) is table

    def test_drops_rows_and_layers_and_narrows_exact_counts(self) -> None:
        large = float(2**24 + 1)
        counts = csr_array(np.array([[1, 0], [0, large], [3, 0]]))
        table = ad.AnnData(counts, obs=pd.DataFrame({"area": [9, 8, 7]}, index=["7", "3", "5"]))
        table.layers["exon"] = csr_array(np.array([[1, 0], [0, 2], [3, 0]], dtype=np.float64))
        assert isinstance(table.X, csr_array)
        assert table.X.dtype == np.float64

        selected = select_cells(table, pd.Index([5, 7], dtype=np.uint64), context=CONTEXT, path=PATH)

        assert selected is not table
        assert selected.obs_names.tolist() == ["7", "5"]
        assert selected.obs["area"].tolist() == [9, 7]
        for matrix, expected in ((selected.X, [[1, 0], [3, 0]]), (selected.layers["exon"], [[1, 0], [3, 0]])):
            assert isinstance(matrix, csr_array)
            assert matrix.dtype == np.float32
            np.testing.assert_array_equal(matrix.toarray(), expected)
        assert table.n_obs == 3


class TestFinalizePositions:
    def test_one_new_coordinate_buffer_and_unchanged_expression(self) -> None:
        native = np.array([[10, 20], [30, 40]], dtype=np.int64)
        table = ad.AnnData(csr_array([[11, 0], [0, 22]]))
        table.obsm["spatial"] = native
        table.layers["counts"] = csr_array([[5, 0], [0, 6]])
        table.raw = table.copy()
        expression, counts, raw = table.X, table.layers["counts"], table.raw

        finalize_positions(
            table, 500, "vendor_cell_center", calibration="Known 500 nm grid", context=CONTEXT, path=PATH
        )

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
        assert table.uns[POSITIONS_KEY] == {
            "schema_version": 1,
            "coordinates": {
                "spatial": {
                    "axes": ["x", "y"],
                    "units": "micrometer",
                    "frame": "global",
                    "position": "vendor_cell_center",
                    "calibration": {"global": "Known 500 nm grid"},
                },
                "spatial_grid": {
                    "axes": ["x", "y"],
                    "units": "grid",
                    "frame": "source_grid",
                    "position": "vendor_cell_center",
                },
            },
        }

    @pytest.mark.parametrize("destination", ["spatial_grid", POSITIONS_KEY])
    def test_collisions_fail_before_mutating_native(self, destination: str) -> None:
        native = np.array([[10, 20]], dtype=np.int64)
        table = ad.AnnData(csr_array([[1]]))
        table.obsm["spatial"] = native
        if destination == "spatial_grid":
            table.obsm[destination] = np.array([[9, 9]])
        else:
            table.uns[destination] = {"unrelated": "preserved"}

        with pytest.raises(ReaderFormatError, match="reserved") as caught:
            finalize_positions(
                table, 500, "vendor_cell_center", calibration="Known 500 nm grid", context=CONTEXT, path=PATH
            )
        assert table.obsm["spatial"] is native
        assert caught.value.path == PATH
        assert caught.value.context == CONTEXT


class TestLinkTable:
    """Final linkage preserves the owned table and validates independent targets."""

    def test_exact_membership_and_single_linkage(self, mocker: MockerFixture) -> None:
        table = ad.AnnData(csr_array([[1, 0], [0, 2]]), obs=pd.DataFrame(index=["a", "b"]))
        target = pd.Index([200, 100], dtype=np.uint64)
        parse = mocker.spy(_linkage.TableModel, "parse")
        result = link_table(table, "cells_centroids", target, context=CONTEXT, path=PATH)

        assert result is table
        assert result.obs["instance_id"].tolist() == [200, 100]
        assert result.obs["instance_id"].dtype == np.uint64
        assert result.obs["region"].cat.categories.tolist() == ["cells_centroids"]
        parse.assert_called_once()

    def test_passes_context_and_path_to_linkage(self, mocker: MockerFixture) -> None:
        table = ad.AnnData(csr_array([[1]]), obs=pd.DataFrame(index=["a"]))
        parse = mocker.patch("spatialdata_io.readers.stereoseq._tables.parse_linked_table", return_value=table)

        link_table(table, "cells_centroids", pd.Index([1]), context=CONTEXT, path=PATH)

        parse.assert_called_once()
        assert parse.call_args.kwargs["context"] == CONTEXT
        assert parse.call_args.kwargs["path"] == PATH

    @pytest.mark.parametrize("column", ["region", "instance_id"])
    def test_conflicting_reserved_annotations(self, column: str) -> None:
        table = ad.AnnData(csr_array([[1]]), obs=pd.DataFrame({column: ["wrong"]}, index=["a"]))

        with pytest.raises(ReaderFormatError, match="reserved") as caught:
            link_table(table, "cells_centroids", pd.Index([1]), context=CONTEXT, path=PATH)
        assert caught.value.path == PATH

    def test_matching_reserved_annotations(self) -> None:
        table = ad.AnnData(csr_array([[1]]), obs=pd.DataFrame({"region": ["cells"], "instance_id": [1]}, index=["a"]))

        assert link_table(table, "cells", pd.Index([1]), context=CONTEXT, path=PATH) is table
