"""Seeker bead identity, geometry and calibrated scale."""

from __future__ import annotations

import numpy as np
import pandas as pd
from spatialdata.transformations import Scale, get_transformation

from spatialdata_io.readers.curio._geometry import make_shapes


class TestMakeShapes:
    def test_native_centres_calibrated_radius_and_independent_bead_index(self) -> None:
        xy = np.array([[2**53 + 1, 0], [7, 3]], dtype=np.int64)
        bead_ids = pd.Index(["007", "a"])
        shapes = make_shapes(xy, bead_ids)
        np.testing.assert_array_equal(shapes.geometry.x, [float(2**53), 7.0])
        np.testing.assert_array_equal(shapes.geometry.y, [0.0, 3.0])
        np.testing.assert_allclose(shapes.radius, [5.0 / 0.647, 5.0 / 0.647])
        transform = get_transformation(shapes, "global")
        assert isinstance(transform, Scale)
        np.testing.assert_allclose(transform.scale, [0.647, 0.647])
        assert shapes.index.tolist() == ["007", "a"]
        shapes.index.name = "bead"
        assert bead_ids.name is None
