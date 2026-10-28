"""Exact vocabulary, definition and error format of the invalid-geometry policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.geometry import (
    invalid_geometry_mask,
    raise_invalid_geometries,
    validate_invalid_geometry_policy,
)

SQUARE = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
# Crossing diagonals: valid ring coordinates, invalid polygon.
BOWTIE = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
# A ring touching itself at one vertex (a "figure eight" pinch).
SELF_TOUCH = Polygon([(0, 0), (4, 0), (4, 4), (2, 0), (0, 4)])
# A hole outside its shell.
HOLE_OUTSIDE = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], holes=[[(5, 5), (6, 5), (6, 6), (5, 6)]])
CONTEXT = ReaderErrorContext("synthetic", "boundaries")


class TestValidateInvalidGeometryPolicy:
    @pytest.mark.parametrize("value", ["keep", "drop", "raise"])
    def test_accepts_every_policy(self, value: str) -> None:
        assert validate_invalid_geometry_policy(value) == value

    @pytest.mark.parametrize("value", ["Keep", "repair", "", "drop "])
    def test_rejects_unknown_values(self, value: str) -> None:
        with pytest.raises(ValueError, match=r"invalid_geometries must be one of \('keep', 'drop', 'raise'\)"):
            validate_invalid_geometry_policy(value)

    @pytest.mark.parametrize("value", [None, True, 1, ["keep"]])
    def test_rejects_non_strings(self, value: object) -> None:
        with pytest.raises(TypeError, match="invalid_geometries must be one of"):
            validate_invalid_geometry_policy(value)


class TestInvalidGeometryMask:
    def test_classifies_valid_and_invalid_polygons(self) -> None:
        geometries = np.array([SQUARE, BOWTIE, SELF_TOUCH, HOLE_OUTSIDE, MultiPolygon([SQUARE])], dtype=object)

        mask = invalid_geometry_mask(geometries)

        np.testing.assert_array_equal(mask, [False, True, True, True, False])
        assert mask.dtype == np.bool_

    def test_definition_is_shapely_validity(self) -> None:
        rng = np.random.default_rng(20261026)
        geometries = np.asarray(shapely.polygons(rng.integers(0, 4, size=(200, 4, 2)).astype(np.float64)), dtype=object)

        mask = invalid_geometry_mask(geometries)

        np.testing.assert_array_equal(mask, ~shapely.is_valid(geometries))
        assert 0 < mask.sum() < len(mask)

    def test_empty_array(self) -> None:
        mask = invalid_geometry_mask(np.array([], dtype=object))
        assert mask.shape == (0,)

    @pytest.mark.parametrize("bad", [None, Polygon()])
    def test_missing_or_empty_geometry_is_a_reader_error(self, bad: Polygon | None) -> None:
        with pytest.raises(ValueError, match="missing or empty"):
            invalid_geometry_mask(np.array([SQUARE, bad], dtype=object))

    def test_requires_one_dimension(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            invalid_geometry_mask(np.array([[SQUARE, LineString([(0, 0), (1, 1)])]], dtype=object))


class TestRaiseInvalidGeometries:
    def test_valid_input_returns(self) -> None:
        raise_invalid_geometries(np.array([False, False]), ["a", "b"], context=CONTEXT)

    def test_reports_count_examples_context_and_path(self) -> None:
        mask = np.zeros(25, dtype=bool)
        mask[3:15] = True
        identifiers = [f"cell-{index}" for index in range(25)]
        path = Path("bundle/cells.parquet")

        with pytest.raises(ReaderFormatError) as caught:
            raise_invalid_geometries(mask, identifiers, context=CONTEXT, path=path)

        error = caught.value
        assert error.context == CONTEXT
        assert error.path == path
        assert error.found == (
            "12 invalid geometries (not shapely.is_valid), e.g. "
            "['cell-3', 'cell-4', 'cell-5', 'cell-6', 'cell-7', 'cell-8', 'cell-9', 'cell-10', 'cell-11', 'cell-12']"
        )
        assert error.expected == "valid geometries; invalid_geometries='keep' preserves them and 'drop' removes them"

    def test_requires_aligned_identifiers(self) -> None:
        with pytest.raises(ValueError, match="equal lengths"):
            raise_invalid_geometries(np.array([True]), ["a", "b"], context=CONTEXT)
