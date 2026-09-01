"""Characterize the current public DBiT workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from spatialdata.models import get_table_keys

from spatialdata_io import __version__
from spatialdata_io.readers.dbit import dbit

if TYPE_CHECKING:
    from pathlib import Path


def _write_layout(path: Path) -> None:
    obs_names = ["CCCCAAAATTGGCCAA", "GGGGTTTTAACCGGTT"]
    table = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[3, 4], [1, 2]], dtype=np.int32)),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=["gene-b", "gene-a"]),
    )
    table.write_h5ad(path / "sample.h5ad")
    (path / "barcode_list.txt").write_text(
        "A1\tAACCGGTT\nA2\tTTGGCCAA\nB1\tGGGGTTTT\nB2\tCCCCAAAA\n",
        encoding="utf-8",
    )


class TestDbit:
    """Characterize identifiers, sparse values, geometry, and linkage."""

    def test_synthetic_layout_preserves_public_contract(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)

        result = dbit(tmp_path, dataset_id="sample", border=True, border_scale=1.0)

        assert list(result.shapes) == ["sample"]
        assert list(result.tables) == ["table"]
        assert not result.images
        assert not result.labels
        assert not result.points
        assert result.attrs == {
            "spatialdata_io_reader": "dbit",
            "spatialdata_io_software_version": __version__,
        }

        shapes = result.shapes["sample"]
        assert shapes.index.tolist() == [0, 1]
        assert shapes.geometry.geom_type.tolist() == ["Polygon", "Polygon"]
        np.testing.assert_allclose(shapes.total_bounds, np.asarray([0.125, 0.125, 1.875, 1.875]))

        table = result.tables["table"]
        assert sparse.isspmatrix_csr(table.X)
        np.testing.assert_array_equal(table.X.toarray(), np.asarray([[1, 2], [3, 4]], dtype=np.int32))
        assert table.obs_names.tolist() == ["GGGGTTTTAACCGGTT", "CCCCAAAATTGGCCAA"]
        assert table.var_names.tolist() == ["gene-b", "gene-a"]
        assert table.obs["array_A"].tolist() == [1, 2]
        assert table.obs["array_B"].tolist() == [1, 2]
        assert table.obs["pixel_id"].tolist() == [0, 1]
        assert get_table_keys(table) == ("sample", "region", "pixel_id")
