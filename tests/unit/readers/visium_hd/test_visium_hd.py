"""Test Visium HD image and projective-transform helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from PIL import Image as PILImage

from spatialdata_io.readers.visium_hd import (
    _decompose_projective_matrix,
    _load_image,
    _projective_matrix_is_affine,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture
    from xarray import DataArray, DataTree

# --- UNIT TESTS FOR HELPER FUNCTIONS ---


def test_projective_matrix_is_affine() -> None:
    """Test the affine matrix check function."""
    # An affine matrix should have [0, 0, 1] as its last row
    affine_matrix = np.array([[2, 0.5, 10], [0.5, 2, 20], [0, 0, 1]])
    assert _projective_matrix_is_affine(affine_matrix)

    # A projective matrix is not affine if the last row is different
    projective_matrix = np.array([[2, 0.5, 10], [0.5, 2, 20], [0.01, 0.02, 1]])
    assert not _projective_matrix_is_affine(projective_matrix)


def test_decompose_projective_matrix() -> None:
    """Test the decomposition of a projective matrix into affine and shift components."""
    projective_matrix = np.array([[1, 2, 3], [4, 5, 6], [0.1, 0.2, 1]])
    affine, shift = _decompose_projective_matrix(projective_matrix)

    expected_affine = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])

    # The affine component should be correctly extracted
    assert np.allclose(affine, expected_affine)
    # Recomposing the affine and shift matrices should yield the original projective matrix
    assert np.allclose(affine @ shift, projective_matrix)


class TestLoadImage:
    """Test Pillow global-state ownership while reading Visium HD images."""

    @pytest.mark.parametrize("fail", [False, True])
    def test_max_image_pixels_and_options_are_restored(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        *,
        fail: bool,
    ) -> None:
        image_path = tmp_path / "image.btf"
        image_path.touch()
        images = cast("dict[str, DataArray | DataTree]", {})
        options = {"MAX_IMAGE_PIXELS": 321, "mode": "I"}
        previous = PILImage.MAX_IMAGE_PIXELS
        reader = mocker.patch("spatialdata_io.readers.visium_hd.imread2")
        if fail:
            reader.side_effect = RuntimeError("injected")
            with pytest.raises(RuntimeError, match="injected"):
                _load_image(image_path, images, "_image", "sample", options, {}, None)
        else:
            reader.return_value = np.zeros((3, 4, 6), dtype=np.uint8)
            parsed = mocker.sentinel.parsed
            mocker.patch("spatialdata_io.readers.visium_hd.Image2DModel.parse", return_value=parsed)
            _load_image(image_path, images, "_image", "sample", options, {}, None)
            assert images == {"sample_image": parsed}

        assert previous == PILImage.MAX_IMAGE_PIXELS
        assert options == {"MAX_IMAGE_PIXELS": 321, "mode": "I"}
        reader.assert_called_once_with(image_path, mode="I")
