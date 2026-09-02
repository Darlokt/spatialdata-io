"""Tests for the public Visium reader boundary."""

from __future__ import annotations

import inspect
from typing import cast

import pytest

import spatialdata_io
from spatialdata_io.readers.visium import visium


class TestVisium:
    """Keep the deliberately small public API and early option validation exact."""

    def test_public_signature_contains_only_format_concerns(self) -> None:
        signature = inspect.signature(visium)

        assert list(signature.parameters) == [
            "path",
            "dataset_id",
            "counts_source",
            "fullres_image_file",
            "positions_file",
            "scale_factors_file",
            "make_var_names_unique",
            "data_axes",
            "tiff_series",
            "chunks",
        ]
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for name, parameter in signature.parameters.items()
            if name != "path"
        )
        assert signature.parameters["make_var_names_unique"].default is True
        assert signature.parameters["data_axes"].default is None
        assert signature.parameters["tiff_series"].default is None
        assert signature.parameters["chunks"].default is None

    def test_lazy_root_export_is_the_same_reader(self) -> None:
        assert spatialdata_io.visium is visium

    def test_rejects_nonboolean_uniqueness_before_filesystem_work(self) -> None:
        with pytest.raises(TypeError, match="make_var_names_unique"):
            visium("missing", make_var_names_unique=cast("bool", 1))
