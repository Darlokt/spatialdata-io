"""CODEX radius-one geometry, source IDs, and coordinate ownership."""

from __future__ import annotations

from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
from spatialdata.transformations import BaseTransformation, get_transformation

from spatialdata_io.readers.codex._geometry import _make_shapes


class TestMakeShapes:
    """Keep cell centers and integer IDs independent of observation names."""

    def test_geometry_and_input_ownership(self) -> None:
        coordinates = np.asarray([[20.0, 1.5], [2.5, 30.0]])
        table = ad.AnnData(obs=pd.DataFrame({"cell_id": [7, 2]}, index=["a", "b"]))
        table.obsm["spatial"] = coordinates

        result = _make_shapes(table)

        assert result.index.tolist() == [7, 2]
        np.testing.assert_array_equal(result.geometry.x, [20, 2.5])
        np.testing.assert_array_equal(result.geometry.y, [1.5, 30])
        np.testing.assert_array_equal(result["radius"], [1, 1])
        np.testing.assert_array_equal(coordinates, [[20, 1.5], [2.5, 30]])
        assert table.obs_names.tolist() == ["a", "b"]
        transform = cast("BaseTransformation", get_transformation(result, to_coordinate_system="global"))
        np.testing.assert_array_equal(transform.to_affine_matrix(("x", "y"), ("x", "y")), np.eye(3))
