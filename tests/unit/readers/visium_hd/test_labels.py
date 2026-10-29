"""Exact table-ordered bin labels in the physical slide frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from spatialdata.transformations import Affine, get_transformation

from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers.visium_hd._labels import make_bin_labels
from spatialdata_io.readers.visium_hd._transforms import SlideFrame

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestMakeBinLabels:
    """Labels and square boundaries share exactly the same physical frame."""

    def test_table_order_background_chunks_and_corners(self) -> None:
        matrix = np.array([[0.0, 1, 3], [1, 0, -5], [0, 0, 1]])
        frame = SlideFrame(np.eye(3), matrix, np.eye(3), 0, 0)
        rows, cols = np.array([2, 0, 1], np.uint32), np.array([2, 1, 0], np.uint32)
        labels = make_bin_labels(frame, rows, cols, (3, 3), 16, options=make_raster_options(chunks=(1, 2, 2)))
        np.testing.assert_array_equal(labels.data.compute(), [[0, 2, 0], [3, 0, 0], [0, 0, 1]])
        assert labels.dtype == np.uint32
        assert labels.dims == ("y", "x")
        np.testing.assert_array_equal(labels.coords["x"], [0.5, 1.5, 2.5])
        np.testing.assert_array_equal(labels.coords["y"], [0.5, 1.5, 2.5])
        assert labels.data.chunks == ((2, 1), (2, 1))
        transform = get_transformation(labels, "global")
        assert isinstance(transform, Affine)
        actual = transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        np.testing.assert_array_equal(actual, matrix @ np.diag([16, 16, 1]))
        np.testing.assert_array_equal(actual @ [2.5, 2.5, 1], matrix @ [40, 40, 1])

    def test_empty(self) -> None:
        frame = SlideFrame(np.eye(3), np.eye(3), np.eye(3), 0, 0)
        empty = np.array([], dtype=np.uint32)
        result = make_bin_labels(frame, empty, empty, (2, 3), 2, options=make_raster_options())
        np.testing.assert_array_equal(result.data.compute(), np.zeros((2, 3), dtype=np.uint32))

    def test_label_capacity(self, mocker: MockerFixture) -> None:
        frame = SlideFrame(np.eye(3), np.eye(3), np.eye(3), 0, 0)
        limit = np.iinfo(np.uint8)
        mocker.patch("spatialdata_io.readers.visium_hd._labels.np.iinfo", return_value=limit)
        indices = np.arange(256, dtype=np.uint32)
        with pytest.raises(ValueError, match="uint32 label capacity"):
            make_bin_labels(frame, indices, indices, (256, 256), 2, options=make_raster_options())
