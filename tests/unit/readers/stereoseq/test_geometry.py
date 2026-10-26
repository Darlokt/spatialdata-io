"""Exact grid points, paired-offset contours and strict topology policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from shapely import Polygon
from spatialdata.transformations import Identity, get_transformation

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.stereoseq._geometry import make_points, make_polygons

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def _border(vertices: list[tuple[int, int]]) -> NDArray[np.int16]:
    border = np.full((1, 32, 2), 32767, dtype=np.int16)
    border[0, : len(vertices)] = vertices
    return border


class TestMakePoints:
    """Keep native numerical precision and independent, unsorted source indexes."""

    def test_preserves_exact_values_and_order(self) -> None:
        coordinates = np.array([[2**32 - 1, -(2**31)], [1, 2]], dtype=np.int64)
        index = pd.Index([200, 100], dtype=np.uint64)
        points = make_points(coordinates, index)
        computed = points.compute()
        assert_array_equal(computed[["x", "y"]], coordinates)
        assert_array_equal(computed.index, index)
        assert computed.x.dtype == np.int64
        assert isinstance(get_transformation(points), Identity)

    def test_empty(self) -> None:
        points = make_points(np.empty((0, 2), dtype=np.int64), pd.Index([], dtype=str))
        assert points.compute().empty


class TestMakePolygons:
    """Decode literal contours without independent component filtering or repair."""

    def test_widens_before_adding_and_preserves_orientation(self, tmp_path: Path) -> None:
        centers = np.array([[2**32 - 1, -(2**31)]], dtype=np.int64)
        border = _border([(-2, -1), (3, -1), (3, 2), (-2, 2)])

        shapes = make_polygons(centers, border, pd.Index([800]), path=tmp_path / "cells.gef")
        expected = np.array(
            [
                [2**32 - 3, -(2**31) - 1],
                [2**32 + 2, -(2**31) - 1],
                [2**32 + 2, -(2**31) + 2],
                [2**32 - 3, -(2**31) + 2],
                [2**32 - 3, -(2**31) - 1],
            ],
            dtype=np.int64,
        )
        polygon = shapes.geometry.iloc[0]
        assert isinstance(polygon, Polygon)
        assert_array_equal(polygon.exterior.coords, expected)
        assert shapes.index.tolist() == [800]
        assert isinstance(get_transformation(shapes), Identity)

    @pytest.mark.parametrize("case", ["single_component", "interior", "too_short"])
    def test_malformed_padding(self, tmp_path: Path, case: str) -> None:
        border = _border([(0, 0), (1, 0), (0, 1)])
        if case == "single_component":
            border[0, 2, 0] = 32767
        elif case == "interior":
            border[0, 1] = 32767
        else:
            border[0, 2] = 32767

        with pytest.raises(ReaderFormatError, match="paired"):
            make_polygons(np.zeros((1, 2), dtype=np.int64), border, pd.Index([7]), path=tmp_path / "cells.gef")

    @pytest.mark.parametrize(
        "vertices",
        [
            [(0, 0), (1, 1), (2, 2)],
            [(0, 0), (1, 1), (1, 0), (0, 1)],
            # The first and closing edges overlap exactly; translation cannot fix this source contour.
            [
                (4, -6),
                (4, -5),
                (2, -3),
                (1, -3),
                (-6, 1),
                (-6, 3),
                (-4, 5),
                (2, 5),
                (3, 4),
                (4, 4),
                (4, 3),
                (6, 1),
                (5, 0),
                (5, -3),
                (4, -4),
            ],
        ],
    )
    def test_invalid_topology(self, tmp_path: Path, vertices: list[tuple[int, int]]) -> None:
        with pytest.raises(ReaderFormatError, match=r"invalid polygon.*55"):
            make_polygons(
                np.array([[4802, 5360]], dtype=np.int64), _border(vertices), pd.Index([55]), path=tmp_path / "cells.gef"
            )

    def test_empty(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="empty cell population"):
            make_polygons(
                np.empty((0, 2), dtype=np.int64),
                np.empty((0, 32, 2), dtype=np.int16),
                pd.Index([], dtype=np.uint64),
                path=tmp_path / "cells.gef",
            )
