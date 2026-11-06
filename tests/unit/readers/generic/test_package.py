"""Tests for the generic reader's public package surface."""

from __future__ import annotations

import inspect

import spatialdata_io
import spatialdata_io.readers.generic as generic_package


class TestGenericPackage:
    """Protect the package and lazy root imports."""

    def test_exports_only_reader_functions(self) -> None:
        assert generic_package.__all__ == ["generic", "geojson", "image"]

    def test_root_exports_are_the_package_functions(self) -> None:
        assert spatialdata_io.generic is generic_package.generic
        assert spatialdata_io.geojson is generic_package.geojson
        assert spatialdata_io.image is generic_package.image

    def test_image_selection_and_raster_options_are_keyword_only(self) -> None:
        parameters = inspect.signature(generic_package.image).parameters

        for name in ("tiff_series", "tiff_level", "chunks", "scale_factors"):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
