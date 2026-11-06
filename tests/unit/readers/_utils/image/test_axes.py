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

    def test_drops_singleton_axes_absent_from_target(self) -> None:
        source = np.arange(3 * 4).reshape(1, 3, 1, 4)
        data = da.from_array(source, chunks=(1, 3, 1, 4))

        result = normalize_raster_axes(data, ("t", "y", "Z", "x"), target_axes=("y", "x"))

        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result.compute(), source[0, :, 0, :])

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
        ("source_axes", "target_axes", "add_missing", "match"),
        [
            (("y",), ("y", "x"), (), "rank"),
            (("y", "y"), ("y", "x"), (), "duplicate"),
            (("c", "y"), ("c", "y"), (), "contain 'x' and 'y'"),
            (("t", "y", "x"), ("y", "x"), (), r"Axis 't' has length 2 and is absent from target_axes"),
            (("y", "x"), ("c", "y", "x"), (), "add_missing"),
            (("y", "x"), ("y", "x"), ("z",), "add_missing contains axes absent from target_axes"),
            (("yy", "x"), ("y", "x"), (), "single alphabetic"),
        ],
    )
    def test_rejects_ambiguous_or_invalid_axes(
        self,
        source_axes: tuple[str, ...],
        target_axes: tuple[str, ...],
        add_missing: tuple[str, ...],
        match: str,
    ) -> None:
        shape = (2, 3) if match == "rank" else (2,) * len(source_axes)
        data = da.zeros(shape, chunks=shape)

        with pytest.raises(ValueError, match=match):
            normalize_raster_axes(data, source_axes, target_axes=target_axes, add_missing=add_missing)
