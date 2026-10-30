"""Exercise asymmetric DBiT grids and the fifty-position protocol boundary."""

from __future__ import annotations

from itertools import islice, product
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from spatialdata.models import get_table_keys

from spatialdata_io import dbit

if TYPE_CHECKING:
    from pathlib import Path


class TestDbit:
    """Keep A/x and B/y distinct through source reordering and final linkage."""

    def test_four_spots_preserve_off_diagonal_axis_and_count_alignment(self, tmp_path: Path) -> None:
        # Asymmetric values and off-diagonal spots expose swapped barcode halves
        # and sorting by the wrong primary axis.
        source_names = [
            "GGGGGGGGCCCCCCCC-topright",
            "TTTTTTTTAAAAAAAA-bottomleft",
            "TTTTTTTTCCCCCCCC-bottomright",
            "GGGGGGGGAAAAAAAA-topleft",
        ]
        source_counts = sparse.csr_array([[21, 210], [12, 120], [22, 220], [11, 110]], dtype=np.uint16)
        ad.AnnData(
            X=source_counts,
            obs=pd.DataFrame({"source_row": [0, 1, 2, 3]}, index=source_names),
            var=pd.DataFrame(index=["gene-z", "gene-a"]),
            layers={"counts": source_counts.copy()},
        ).write_h5ad(tmp_path / "asymmetric.h5ad")
        (tmp_path / "barcode_list.txt").write_text(
            "B2\tTTTTTTTT\nA1\tAAAAAAAA\nB1\tGGGGGGGG\nA2\tCCCCCCCC\n",
            encoding="utf-8",
        )

        result = dbit(tmp_path)

        table = result.tables["table"]
        assert table.obs_names.tolist() == [
            "GGGGGGGGAAAAAAAA-topleft",
            "TTTTTTTTAAAAAAAA-bottomleft",
            "GGGGGGGGCCCCCCCC-topright",
            "TTTTTTTTCCCCCCCC-bottomright",
        ]
        assert table.var_names.tolist() == ["gene-z", "gene-a"]
        assert table.obs["source_row"].tolist() == [3, 1, 0, 2]
        assert table.obs["array_A"].tolist() == [1, 1, 2, 2]
        assert table.obs["array_B"].tolist() == [1, 2, 1, 2]
        np.testing.assert_array_equal(table.obsm["spatial_grid"], [[1, 1], [1, 2], [2, 1], [2, 2]])

        assert isinstance(table.X, sparse.csr_array)
        assert table.X.dtype == np.uint16
        np.testing.assert_array_equal(table.X.toarray(), [[11, 110], [12, 120], [21, 210], [22, 220]])
        counts = table.layers["counts"]
        assert isinstance(counts, sparse.csr_array)
        assert counts.dtype == np.uint16
        np.testing.assert_array_equal(counts.toarray(), [[11, 110], [12, 120], [21, 210], [22, 220]])

        grid = result.shapes["asymmetric"]
        np.testing.assert_allclose(
            grid.bounds.to_numpy(),
            [
                [0.125, 0.125, 0.875, 0.875],
                [0.125, 1.125, 0.875, 1.875],
                [1.125, 0.125, 1.875, 0.875],
                [1.125, 1.125, 1.875, 1.875],
            ],
        )

        assert get_table_keys(table) == ("asymmetric", "region", "pixel_id")
        assert table.obs["pixel_id"].tolist() == grid.index.tolist() == [0, 1, 2, 3]

    def test_fifty_position_map_preserves_extreme_grid_locations(self, tmp_path: Path) -> None:
        sequences = ["".join(bases) for bases in islice(product("ACGT", repeat=8), 100)]
        a = sequences[:50]
        b = sequences[50:]
        barcode_rows = [f"B{position}\t{b[position - 1]}\n" for position in range(50, 0, -1)]
        barcode_rows.extend(f"A{position}\t{a[position - 1]}\n" for position in range(50, 0, -1))
        (tmp_path / "barcode_list.txt").write_text("".join(barcode_rows), encoding="utf-8")

        source_names = [b[0] + a[49], b[49] + a[0]]
        ad.AnnData(
            X=sparse.csr_array([[501], [150]], dtype=np.int32),
            obs=pd.DataFrame(index=source_names),
            var=pd.DataFrame(index=["gene"]),
        ).write_h5ad(tmp_path / "maximum.h5ad")

        result = dbit(tmp_path, include_border=False)

        table = result.tables["table"]
        assert table.obs_names.tolist() == [source_names[1], source_names[0]]
        np.testing.assert_array_equal(table.obsm["spatial_grid"], [[1, 50], [50, 1]])

        assert isinstance(table.X, sparse.csr_array)
        assert table.X.dtype == np.int32
        np.testing.assert_array_equal(table.X.toarray(), [[150], [501]])

        grid = result.shapes["maximum"]
        np.testing.assert_allclose(grid.bounds.to_numpy(), [[0, 49, 1, 50], [49, 0, 50, 1]])
        assert get_table_keys(table) == ("maximum", "region", "pixel_id")
        assert table.obs["pixel_id"].tolist() == grid.index.tolist() == [0, 1]
