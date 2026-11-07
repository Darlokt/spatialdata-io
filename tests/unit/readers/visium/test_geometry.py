"""Tests for Visium spot geometry and transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from spatialdata._core.validation import check_valid_name
from spatialdata.transformations import BaseTransformation, get_transformation

from spatialdata_io.readers.visium._geometry import make_circles, make_frames
from spatialdata_io.readers.visium._metadata import ScaleFactors

if TYPE_CHECKING:
    from geopandas import GeoDataFrame


def _matrix(element: GeoDataFrame, coordinate_system: str) -> np.ndarray:
    transformation = get_transformation(element, to_coordinate_system=coordinate_system)
    assert isinstance(transformation, BaseTransformation)
    return np.asarray(transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")))


class TestMakeFrames:
    """Name the pixel frames and construct exact full-resolution-to-PNG scales."""

    def test_names_and_uses_xy_axis_order(self) -> None:
        frames = make_frames("sample", ScaleFactors(0.5, 0.25, 6.0, {}))

        assert (frames.fullres, frames.hires, frames.lowres) == (
            "sample_pixels",
            "sample_hires_pixels",
            "sample_lowres_pixels",
        )
        np.testing.assert_array_equal(
            frames.hires_scale.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
            np.diag([0.5, 0.5, 1.0]),
        )
        np.testing.assert_array_equal(
            frames.lowres_scale.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
            np.diag([0.25, 0.25, 1.0]),
        )

    @pytest.mark.parametrize(
        "dataset_id", ["sample", "a", "Table_", "x_hires", "x_pixels", "S_lowres", "unnamed", "V1_Mouse_Brain", "a.b-c"]
    )
    def test_frames_never_collide_with_each_other_or_elements(self, dataset_id: str) -> None:
        frames = make_frames(dataset_id, ScaleFactors(0.5, 0.25, 6.0, {}))
        names = [frames.fullres, frames.hires, frames.lowres]
        elements = [dataset_id, "table", *(f"{dataset_id}_{kind}_image" for kind in ("full", "hires", "lowres"))]

        lowered = [name.lower() for name in names + elements]
        assert len(set(lowered)) == len(lowered)
        for name in names:
            check_valid_name(name)


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
        frames = make_frames("sample", ScaleFactors(0.5, 0.25, 6.0, {}))

        circles = make_circles(positions, radius=3.0, frames=frames)

        assert circles.index.tolist() == ["a", "b"]
        assert circles.index.name == "spot_id"
        assert positions.index.name == "barcode"
        np.testing.assert_array_equal(circles.geometry.x, np.asarray([20.0, 40.0]))
        np.testing.assert_array_equal(circles.geometry.y, np.asarray([10.0, 30.0]))
        np.testing.assert_array_equal(circles["radius"], np.asarray([3.0, 3.0]))

        transformations = get_transformation(circles, get_all=True)
        assert isinstance(transformations, dict)
        assert set(transformations) == {
            "sample_pixels",
            "sample_hires_pixels",
            "sample_lowres_pixels",
        }
        np.testing.assert_array_equal(_matrix(circles, "sample_pixels"), np.eye(3))
        np.testing.assert_array_equal(_matrix(circles, "sample_hires_pixels"), np.diag([0.5, 0.5, 1.0]))
        np.testing.assert_array_equal(_matrix(circles, "sample_lowres_pixels"), np.diag([0.25, 0.25, 1.0]))
