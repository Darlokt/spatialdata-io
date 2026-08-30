"""Tests for the generic reader's public package surface."""

from __future__ import annotations

import inspect

import spatialdata_io.readers.generic as generic_package


class TestGenericPackage:
    """Protect public imports after the module-to-package conversion."""

    def test_exports_only_intentional_public_names(self) -> None:
        assert generic_package.__all__ == [
            "VALID_IMAGE_TYPES",
            "VALID_SHAPE_TYPES",
            "generic",
            "geojson",
            "image",
        ]

    def test_exposes_supported_suffixes(self) -> None:
        assert generic_package.VALID_IMAGE_TYPES == (
            ".btf",
            ".tif",
            ".tiff",
            ".png",
            ".jpeg",
            ".jpg",
        )
        assert generic_package.VALID_SHAPE_TYPES == (".geojson",)

    def test_image_options_are_keyword_only_and_backend_free(self) -> None:
        parameters = inspect.signature(generic_package.image).parameters

        assert "use_tiff_memmap" not in parameters
        for name in ("tiff_series", "tiff_level", "chunks", "scale_factors"):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
