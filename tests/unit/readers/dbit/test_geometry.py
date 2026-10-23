"""Verify DBiT polygon vertex order, inset, scale, and numerical validity."""

from __future__ import annotations

import numpy as np
import pytest
from shapely import Polygon
from spatialdata.transformations import Identity, get_transformation

from spatialdata_io.readers.dbit._geometry import _make_grid


class TestMakeGrid:
    """Keep A/x and B/y square geometry exact across border policies."""

    @pytest.mark.parametrize("scale", [1.0, 5.0, 1.3])
    @pytest.mark.parametrize(("include_border", "border_scale"), [(False, 1.0), (True, 0.0), (True, 1.0), (True, 2.0)])
    def test_vertices_and_membership(self, scale: float, *, include_border: bool, border_scale: float) -> None:
        coordinates = np.array([[1, 2], [3, 1]], dtype=np.int64)
        original = coordinates.copy()

        grid = _make_grid(coordinates, scale=scale, include_border=include_border, border_scale=border_scale)

        inset = 0.125 * (scale * border_scale) if include_border else 0.0
        polygon = grid.geometry.iloc[0]
        assert isinstance(polygon, Polygon)
        np.testing.assert_array_equal(
            polygon.exterior.coords,
            [
                [inset, scale + inset],
                [inset, 2 * scale - inset],
                [scale - inset, 2 * scale - inset],
                [scale - inset, scale + inset],
                [inset, scale + inset],
            ],
        )
        assert grid.index.tolist() == [0, 1]
        assert grid.is_valid.all()
        np.testing.assert_allclose(grid.area, (scale - 2 * inset) ** 2, rtol=1e-14)
        np.testing.assert_array_equal(coordinates, original)
        assert get_transformation(grid, get_all=True) == {"global": Identity()}

    def test_rejects_width_lost_to_float_rounding(self) -> None:
        with pytest.raises(ValueError, match="representable positive square width"):
            _make_grid(
                np.array([[50, 50]], dtype=np.int64),
                scale=1.0,
                include_border=True,
                border_scale=float(np.nextafter(4.0, 0.0)),
            )
