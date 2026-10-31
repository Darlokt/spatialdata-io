"""CODEX radius-one geometry, source IDs, and coordinate ownership."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from spatialdata.transformations import BaseTransformation, Identity, get_transformation

from spatialdata_io.readers.codex._geometry import _make_shapes


class TestMakeShapes:
    """Keep cell centers and integer IDs independent of observation names."""

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_geometry_and_input_ownership(self, dtype: type[np.integer]) -> None:
        coordinates = np.asarray([[20.0, 1.5], [2.5, 30.0]])
        identifiers = pd.Index(np.asarray([7, 2], dtype=dtype), name="cell_id")

        result = _make_shapes(identifiers, coordinates, frame="table_pixels", transformation=Identity())

        assert result.index.tolist() == [7, 2]
        np.testing.assert_array_equal(result.geometry.x, [20, 2.5])
        np.testing.assert_array_equal(result.geometry.y, [1.5, 30])
        np.testing.assert_array_equal(result["radius"], [1, 1])
        np.testing.assert_array_equal(coordinates, [[20, 1.5], [2.5, 30]])
        pd.testing.assert_index_equal(result.index, identifiers)
        assert identifiers.tolist() == [7, 2]
        transform = cast("BaseTransformation", get_transformation(result, to_coordinate_system="table_pixels"))
        np.testing.assert_array_equal(transform.to_affine_matrix(("x", "y"), ("x", "y")), np.eye(3))
