"""Tests for Visium spot geometry and transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from spatialdata.transformations import BaseTransformation, get_transformation

from spatialdata_io.readers.visium._geometry import make_circles, make_transforms
from spatialdata_io.readers.visium._metadata import ScaleFactors

if TYPE_CHECKING:
    from geopandas import GeoDataFrame


def _matrix(element: GeoDataFrame, coordinate_system: str) -> np.ndarray:
    transformation = get_transformation(element, to_coordinate_system=coordinate_system)
    assert isinstance(transformation, BaseTransformation)
    return np.asarray(transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")))


class TestMakeTransforms:
    """Construct exact full-resolution and image-level scale mappings."""

    def test_uses_xy_axis_order(self) -> None:
        transforms = make_transforms(ScaleFactors(0.5, 0.25, 6.0, {}))

        np.testing.assert_array_equal(
            transforms.hires.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
            np.diag([0.5, 0.5, 1.0]),
        )
        np.testing.assert_array_equal(
            transforms.lowres.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
            np.diag([0.25, 0.25, 1.0]),
        )


class TestMakeCircles:
    """Preserve barcode identity and construct radius rather than diameter."""

    def test_constructs_barcode_indexed_fullres_pixel_circles(self) -> None:
        positions = pd.DataFrame(
            {
                "pxl_col_in_fullres": [20.0, 40.0],
                "pxl_row_in_fullres": [10.0, 30.0],
            },
            index=pd.Index(["a", "b"], name="barcode"),
        )
        transforms = make_transforms(ScaleFactors(0.5, 0.25, 6.0, {}))

        circles = make_circles(positions, dataset_id="sample", radius=3.0, transforms=transforms)

        assert circles.index.tolist() == ["a", "b"]
        assert circles.index.name == "spot_id"
        np.testing.assert_array_equal(circles.geometry.x, np.asarray([20.0, 40.0]))
        np.testing.assert_array_equal(circles.geometry.y, np.asarray([10.0, 30.0]))
        np.testing.assert_array_equal(circles["radius"], np.asarray([3.0, 3.0]))
        np.testing.assert_array_equal(_matrix(circles, "sample"), np.eye(3))
        np.testing.assert_array_equal(_matrix(circles, "sample_downscaled_hires"), np.diag([0.5, 0.5, 1.0]))
        np.testing.assert_array_equal(_matrix(circles, "sample_downscaled_lowres"), np.diag([0.25, 0.25, 1.0]))
