"""Unit tests for generic input orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers.generic._reader import generic

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestGeneric:
    """Validate format routing and option ownership."""

    def test_routes_images_with_normalized_defaults(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        path = tmp_path / "IMAGE.TIF"
        path.touch()
        expected = object()
        parse = mocker.patch(
            "spatialdata_io.readers.generic._reader.image",
            return_value=expected,
        )

        result = generic(path, data_axes="YXC")

        assert result is expected
        parse.assert_called_once_with(
            path,
            data_axes="YXC",
            coordinate_system="global",
            tiff_series=0,
            tiff_level=0,
            chunks=None,
        )

    def test_routes_geojson_without_image_options(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        path = tmp_path / "shapes.GEOJSON"
        path.touch()
        expected = object()
        parse = mocker.patch(
            "spatialdata_io.readers.generic._reader.geojson",
            return_value=expected,
        )

        result = generic(path, coordinate_system="cells")

        assert result is expected
        parse.assert_called_once_with(path, coordinate_system="cells")

    @pytest.mark.parametrize(
        ("data_axes", "tiff_series", "match"),
        [
            (None, 0, "data_axes must be provided"),
            ("yx", 1, "only select TIFF"),
        ],
    )
    def test_rejects_invalid_png_options(
        self,
        tmp_path: Path,
        data_axes: str | None,
        tiff_series: int,
        match: str,
    ) -> None:
        path = tmp_path / "image.png"
        path.touch()

        with pytest.raises(ValueError, match=match):
            generic(path, data_axes=data_axes, tiff_series=tiff_series)

    @pytest.mark.parametrize(
        ("data_axes", "tiff_level", "chunks", "match"),
        [
            ("yx", 0, None, "cannot be used with GeoJSON"),
            (None, 1, None, "cannot be used with GeoJSON"),
            (None, 0, 256, "cannot be used with GeoJSON"),
        ],
    )
    def test_rejects_image_options_for_geojson(
        self,
        tmp_path: Path,
        data_axes: str | None,
        tiff_level: int,
        chunks: int | None,
        match: str,
    ) -> None:
        path = tmp_path / "shapes.geojson"
        path.touch()

        with pytest.raises(ValueError, match=match):
            generic(path, data_axes=data_axes, tiff_level=tiff_level, chunks=chunks)

    def test_rejects_unknown_suffix(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported input suffix"):
            generic(tmp_path / "image.bmp", data_axes="yxc")
