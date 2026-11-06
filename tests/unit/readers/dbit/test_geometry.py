"""Verify DBiT polygon vertex order, inset, transformations and numerical validity."""

from __future__ import annotations

import numpy as np
import pytest
from shapely import Polygon
from spatialdata.transformations import Identity, Scale, get_transformation

from spatialdata_io.readers.dbit._geometry import make_grid

_IDENTITY = {"grid_pixels": Identity()}


class TestMakeGrid:
    """Keep A/x and B/y square geometry exact in grid units across border policies."""

    @pytest.mark.parametrize(("include_border", "border_scale"), [(False, 1.0), (True, 0.0), (True, 1.0), (True, 2.0)])
    def test_vertices_and_membership(self, *, include_border: bool, border_scale: float) -> None:
        coordinates = np.array([[1, 2], [3, 1]], dtype=np.int64)
        original = coordinates.copy()

        grid = make_grid(
            coordinates, include_border=include_border, border_scale=border_scale, transformations=_IDENTITY
        )

        inset = 0.125 * border_scale if include_border else 0.0
        polygon = grid.geometry.iloc[0]
        assert isinstance(polygon, Polygon)
        np.testing.assert_array_equal(
            polygon.exterior.coords,
            [
                [inset, 1 + inset],
                [inset, 2 - inset],
                [1 - inset, 2 - inset],
                [1 - inset, 1 + inset],
                [inset, 1 + inset],
            ],
        )
        assert grid.index.tolist() == [0, 1]
        assert grid.index.name == "pixel_id"
        assert grid.is_valid.all()
        np.testing.assert_allclose(grid.area, (1 - 2 * inset) ** 2, rtol=1e-14)
        np.testing.assert_array_equal(coordinates, original)
        assert get_transformation(grid, get_all=True) == _IDENTITY

    def test_transformations_are_attached_unchanged(self) -> None:
        transformations = {"grid_pixels": Identity(), "image_pixels": Scale([2.5, 2.5], axes=("x", "y"))}

        grid = make_grid(
            np.array([[1, 1]], dtype=np.int64), include_border=False, border_scale=1.0, transformations=transformations
        )

        assert get_transformation(grid, get_all=True) == transformations
        np.testing.assert_array_equal(grid.total_bounds, [0, 0, 1, 1])

    def test_rejects_width_lost_to_float_rounding(self) -> None:
        with pytest.raises(ValueError, match="representable positive square width"):
            make_grid(
                np.array([[50, 50]], dtype=np.int64),
                include_border=True,
                border_scale=float(np.nextafter(4.0, 0.0)),
                transformations=_IDENTITY,
            )
