"""Characterize the public Visium and CLI signatures."""

from __future__ import annotations

import inspect
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image as PILImage

from spatialdata_io.__main__ import visium_wrapper
from spatialdata_io.readers.visium import _read_image, visium

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestVisium:
    """Characterize the public Visium entry points before migration."""

    def test_public_and_cli_signature_baseline(self) -> None:
        reader_signature = inspect.signature(visium)
        assert list(reader_signature.parameters) == [
            "path",
            "dataset_id",
            "counts_file",
            "fullres_image_file",
            "tissue_positions_file",
            "scalefactors_file",
            "var_names_make_unique",
            "imread_kwargs",
            "image_models_kwargs",
            "kwargs",
        ]
        assert reader_signature.parameters["dataset_id"].default is None
        assert reader_signature.parameters["counts_file"].default == "filtered_feature_bc_matrix.h5"
        assert reader_signature.parameters["var_names_make_unique"].default is True
        assert dict(reader_signature.parameters["imread_kwargs"].default) == {}
        assert dict(reader_signature.parameters["image_models_kwargs"].default) == {}

        callback = visium_wrapper.callback
        assert callback is not None
        wrapper_signature = inspect.signature(callback)
        assert list(wrapper_signature.parameters) == [
            "input",
            "output",
            "dataset_id",
            "counts_file",
            "fullres_image_file",
            "tissue_positions_file",
            "scalefactors_file",
            "var_names_make_unique",
            "imread_kwargs",
            "image_models_kwargs",
        ]
        assert wrapper_signature.parameters["dataset_id"].default is None
        assert wrapper_signature.parameters["counts_file"].default == "filtered_feature_bc_matrix.h5"
        assert wrapper_signature.parameters["var_names_make_unique"].default is True
        assert wrapper_signature.parameters["imread_kwargs"].default == "{}"
        assert wrapper_signature.parameters["image_models_kwargs"].default == "{}"


class TestReadImage:
    """Test Pillow global-state ownership while reading Visium images."""

    @pytest.mark.parametrize("fail", [False, True])
    def test_max_image_pixels_and_options_are_restored(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        *,
        fail: bool,
    ) -> None:
        image_path = tmp_path / "image.btf"
        options = MappingProxyType({"MAX_IMAGE_PIXELS": 123, "mode": "I"})
        previous = PILImage.MAX_IMAGE_PIXELS
        reader = mocker.patch("spatialdata_io.readers.visium.imread2")
        if fail:
            reader.side_effect = RuntimeError("injected")
            with pytest.raises(RuntimeError, match="injected"):
                _read_image(image_path, options)
        else:
            reader.return_value = np.zeros((2, 4, 6), dtype=np.uint8)
            result = _read_image(image_path, options)
            assert result.shape == (2, 4, 6)

        assert previous == PILImage.MAX_IMAGE_PIXELS
        assert dict(options) == {"MAX_IMAGE_PIXELS": 123, "mode": "I"}
        reader.assert_called_once_with(image_path, mode="I")
