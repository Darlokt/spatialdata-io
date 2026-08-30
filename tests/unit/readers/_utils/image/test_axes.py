"""Tests for explicit lazy raster-axis normalization."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
from dask import delayed

from spatialdata_io.readers._utils.image import normalize_raster_axes


class TestNormalizeRasterAxes:
    """Validate the complete public axis-normalization contract."""

    def test_transposes_case_insensitive_axes_lazily(self) -> None:
        source = np.arange(2 * 3 * 5).reshape(2, 3, 5)
        data = da.from_array(source, chunks=(1, 2, 3))

        result = normalize_raster_axes(data, ("C", "Y", "X"), target_axes=("x", "c", "y"))

        assert result.shape == (5, 2, 3)
        np.testing.assert_array_equal(result.compute(), source.transpose(2, 0, 1))

    def test_selects_non_singleton_axes_and_drops_singletons(self) -> None:
        source = np.arange(2 * 1 * 3 * 4).reshape(2, 1, 3, 4)
        data = da.from_array(source, chunks=(1, 1, 3, 4))

        result = normalize_raster_axes(
            data,
            ("t", "z", "y", "x"),
            target_axes=("y", "x"),
            selectors={"T": 1},
        )

        np.testing.assert_array_equal(result.compute(), source[1, 0])

    def test_adds_only_explicit_missing_singleton_axes(self) -> None:
        source = np.arange(12).reshape(3, 4)
        data = da.from_array(source, chunks=(2, 2))

        result = normalize_raster_axes(
            data,
            ("y", "x"),
            target_axes=("c", "y", "x"),
            add_missing={"c"},
        )

        assert result.shape == (1, 3, 4)
        np.testing.assert_array_equal(result.compute(), source[None])

    def test_does_not_execute_source_task(self) -> None:
        @delayed
        def fail_if_computed() -> np.ndarray:
            message = "axis normalization computed pixels"
            raise AssertionError(message)

        data = da.from_delayed(fail_if_computed(), shape=(3, 4), dtype=np.uint8)

        result = normalize_raster_axes(
            data,
            ("y", "x"),
            target_axes=("c", "y", "x"),
            add_missing={"c"},
        )

        assert result.shape == (1, 3, 4)

    @pytest.mark.parametrize(
        ("source_axes", "target_axes", "selectors", "add_missing", "match"),
        [
            (("y",), ("y", "x"), None, (), "rank"),
            (("y", "y"), ("y", "x"), None, (), "duplicate"),
            (("c", "y"), ("c", "y"), None, (), "contain 'x' and 'y'"),
            (("t", "y", "x"), ("y", "x"), None, (), "explicit selector"),
            (("t", "y", "x"), ("y", "x"), {"t": 2}, (), "valid range"),
            (("y", "x"), ("c", "y", "x"), None, (), "add_missing"),
            (("y", "x"), ("y", "x"), {"z": 0}, (), "absent from source_axes"),
            (("t", "y", "x"), ("y", "x"), {"t": 0, "T": 0}, (), "duplicate"),
            (("y", "x"), ("y", "x"), {"y": 0}, (), "omitted from target_axes"),
            (("y", "x"), ("y", "x"), None, ("z",), "absent from target_axes"),
            (("yy", "x"), ("y", "x"), None, (), "single alphabetic"),
        ],
    )
    def test_rejects_ambiguous_or_invalid_axes(
        self,
        source_axes: tuple[str, ...],
        target_axes: tuple[str, ...],
        selectors: dict[str, int] | None,
        add_missing: tuple[str, ...],
        match: str,
    ) -> None:
        shape = (2, 3) if match == "rank" else (2,) * len(source_axes)
        data = da.zeros(shape, chunks=shape)

        with pytest.raises(ValueError, match=match):
            normalize_raster_axes(
                data,
                source_axes,
                target_axes=target_axes,
                selectors=selectors,
                add_missing=add_missing,
            )
