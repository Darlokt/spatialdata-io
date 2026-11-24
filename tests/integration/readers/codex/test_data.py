"""Check the published Akoya tonsil subset against its source values."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from spatialdata_io import codex

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class TestCodexDataset:
    def test_tonsil_cell_measurements(self, require_test_dataset: Callable[[str], Path]) -> None:
        root = require_test_dataset("codex_tonsil")
        with (root / "tonsil-akoya-2018-500.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        features = [col for col in rows[0] if col.startswith("cyc")]
        expected = sparse.csr_array([[float(row[col]) for col in features] for row in rows])

        sdata = codex(root)
        table = sdata.tables["table"]
        assert table.shape == (500, 32)
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.dtype == np.float64
        assert table.var_names.tolist() == [col.split(":")[0] for col in features]
        assert table.var["marker"].tolist() == [col.split(":")[1] for col in features]
        np.testing.assert_array_equal(table.X.indptr, expected.indptr)
        np.testing.assert_array_equal(table.X.indices, expected.indices)
        np.testing.assert_array_equal(table.X.data, expected.data)
        ids = np.array([int(row["cell_id:cell_id"]) for row in rows], dtype=np.uint64)
        assert table.obs["cell_id"].dtype == np.uint64
        np.testing.assert_array_equal(table.obs["cell_id"], ids)
        circles = sdata.shapes["1"]
        np.testing.assert_array_equal(circles.index, ids)
        np.testing.assert_array_equal(circles.geometry.x, [float(row["x:x"]) for row in rows])
        np.testing.assert_array_equal(circles.geometry.y, [float(row["y:y"]) for row in rows])
        np.testing.assert_array_equal(circles["radius"], np.ones(len(ids)))
