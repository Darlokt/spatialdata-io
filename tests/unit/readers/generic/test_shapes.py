"""Unit tests for generic GeoJSON model construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata.transformations import Identity

from spatialdata_io.readers.generic._shapes import geojson

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestGeojson:
    """Validate the shape-model boundary."""

    def test_parses_path_with_owned_transformation(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        path = tmp_path / "shapes.geojson"
        expected = object()
        parse = mocker.patch(
            "spatialdata_io.readers.generic._shapes.ShapesModel.parse",
            return_value=expected,
        )

        result = geojson(path, coordinate_system="cells")

        assert result is expected
        parse.assert_called_once()
        args, kwargs = parse.call_args
        assert args == (path,)
        assert isinstance(kwargs["transformations"]["cells"], Identity)
