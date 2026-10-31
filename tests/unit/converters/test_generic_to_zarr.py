"""Validate Generic conversion options before filesystem access."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from spatialdata_io.converters.generic_to_zarr import generic_to_zarr

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


@pytest.mark.generic
class TestGenericToZarr:
    @pytest.mark.parametrize("coordinate_system", ["", " ", "\t\n", 1])
    def test_rejects_invalid_frame_before_reading_store_or_source(
        self, tmp_path: Path, mocker: MockerFixture, coordinate_system: object
    ) -> None:
        output = tmp_path / "existing.zarr"
        output.mkdir()
        read = mocker.patch("spatialdata_io.converters.generic_to_zarr.SpatialData.read")
        parse = mocker.patch("spatialdata_io.converters.generic_to_zarr.generic")
        error = ValueError if isinstance(coordinate_system, str) else TypeError

        with pytest.raises(error, match="coordinate_system"):
            generic_to_zarr(tmp_path / "missing.geojson", output, coordinate_system=cast("str", coordinate_system))

        read.assert_not_called()
        parse.assert_not_called()
