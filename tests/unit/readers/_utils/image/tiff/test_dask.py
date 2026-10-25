"""Tests for the chunk geometry of TIFF source tasks."""

from __future__ import annotations

import numpy as np
import pytest

from spatialdata_io.readers._utils.image.tiff._dask import STRIP_BAND_BYTES, _task_chunks
from spatialdata_io.readers._utils.image.tiff._types import TiffLevelMetadata


def _level(
    shape: tuple[int, ...], native_chunks: tuple[int, ...], axes: str, dtype: str = "uint16"
) -> TiffLevelMetadata:
    return TiffLevelMetadata(shape=shape, dtype=np.dtype(dtype), axes=tuple(axes), native_chunks=native_chunks)


class TestTaskChunks:
    """Group whole-row chunks into bounded bands and keep every other layout."""

    def test_one_row_strips_reach_the_band_size(self) -> None:
        # A 4096-pixel uint16 row is 8 KiB, so 2048 rows fill one band.
        assert _task_chunks(_level((5000, 4096), (1, 4096), "YX")) == (2048, 4096)
        assert STRIP_BAND_BYTES == 2048 * 4096 * 2

    def test_bands_are_whole_multiples_of_the_strip_height(self) -> None:
        # 7-row strips just under 1 MiB group into whole strips only: 16 strips, 112 rows.
        assert _task_chunks(_level((1000, 149796), (7, 149796), "YX", "uint8")) == (112, 149796)

    def test_strips_larger_than_a_band_stay_single_strips(self) -> None:
        assert _task_chunks(_level((5000, 4096), (2048, 4096), "YX", "float64")) == (2048, 4096)

    @pytest.mark.parametrize(
        ("shape", "native_chunks", "axes", "expected"),
        [
            pytest.param((3, 19, 23), (1, 5, 23), "CYX", (1, 19, 23), id="strips"),
            pytest.param((10, 300, 300), (1, 300, 300), "ZYX", (1, 300, 300), id="one-strip-planes"),
        ],
    )
    def test_band_is_clamped_to_the_plane_height(
        self, shape: tuple[int, ...], native_chunks: tuple[int, ...], axes: str, expected: tuple[int, ...]
    ) -> None:
        assert _task_chunks(_level(shape, native_chunks, axes)) == expected

    def test_contiguous_samples_follow_the_row_axis(self) -> None:
        assert _task_chunks(_level((5000, 4096, 3), (1, 4096, 3), "YXS", "uint8")) == (1365, 4096, 3)

    def test_tiles_as_wide_as_the_image_form_bands(self) -> None:
        assert _task_chunks(_level((5000, 4096), (16, 4096), "YX")) == (2048, 4096)

    @pytest.mark.parametrize(
        ("shape", "native_chunks", "axes"),
        [
            pytest.param((300, 300), (256, 256), "YX", id="tiles"),
            pytest.param((300, 300, 3), (256, 300, 1), "YXS", id="partial-samples"),
            pytest.param((4, 300), (1, 300), "XY", id="row-axis-last"),
            pytest.param((4, 300), (1, 300), "QQ", id="no-row-axis"),
        ],
    )
    def test_keeps_native_chunks_without_whole_rows(
        self, shape: tuple[int, ...], native_chunks: tuple[int, ...], axes: str
    ) -> None:
        assert _task_chunks(_level(shape, native_chunks, axes)) == native_chunks
