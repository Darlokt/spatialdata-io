"""Shared pyramid and chunk options with central defaults."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
from dask import delayed

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS, DEFAULT_TILE_SIZE
from spatialdata_io.readers._utils.image import make_raster_options


class TestMakeRasterOptions:
    """Options validate once, own the caller's factors and give explicit model chunks."""

    def test_defaults(self) -> None:
        options = make_raster_options()

        assert DEFAULT_TILE_SIZE == 1024
        assert DEFAULT_CHUNKS == (1, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE)
        assert DEFAULT_SCALE_FACTORS == ()
        assert options.scale_factors == ()
        assert options.chunks == (1, 1024, 1024)
        assert options.model_scale_factors is None
        assert options.image_chunks() == {"c": 1, "y": DEFAULT_TILE_SIZE, "x": DEFAULT_TILE_SIZE}
        assert options.label_chunks() == {"y": DEFAULT_TILE_SIZE, "x": DEFAULT_TILE_SIZE}

    def test_requested_values(self) -> None:
        factors = [2, 4]

        options = make_raster_options(factors, (3, 256, 512))
        factors.append(8)

        assert options.scale_factors == (2, 4)
        assert options.model_scale_factors == [2, 4]
        # Non-square and channel-grouped chunks are valid explicit requests.
        assert options.image_chunks() == {"c": 3, "y": 256, "x": 512}
        assert options.label_chunks() == {"y": 256, "x": 512}
        # One integer size has the same single tuple representation.
        assert make_raster_options((), 64) == make_raster_options((), (64, 64, 64))
        assert make_raster_options((), 64).chunks == (64, 64, 64)

    @pytest.mark.parametrize(
        ("scale_factors", "chunks", "error", "match"),
        [
            ("22", DEFAULT_CHUNKS, TypeError, "sequence"),
            (2, DEFAULT_CHUNKS, TypeError, "sequence"),
            ([2.0], DEFAULT_CHUNKS, TypeError, "integers"),
            ([True], DEFAULT_CHUNKS, TypeError, "integers"),
            ([1], DEFAULT_CHUNKS, ValueError, "at least 2"),
            ((), 0, ValueError, "positive"),
            ((), (1, 2), ValueError, "three"),
            ((), (1, 0, 2), ValueError, "positive"),
            ((), True, TypeError, "positive integer"),
            ((), (1, 2.0, 3), TypeError, "positive integer"),
            ((), [1, 2, 3], TypeError, "positive integer"),
            ((), None, TypeError, "positive integer"),
        ],
        ids=[
            "text",
            "scalar",
            "float-factor",
            "bool-factor",
            "factor-one",
            "zero",
            "two-sizes",
            "zero-size",
            "bool-chunks",
            "float-size",
            "list",
            "none",
        ],
    )
    def test_invalid_options(self, scale_factors: object, chunks: object, error: type[Exception], match: str) -> None:
        with pytest.raises(error, match=match):
            make_raster_options(scale_factors, chunks)  # pyrefly: ignore[bad-argument-type] - Invalid runtime inputs.


class TestValidateImagePyramid:
    """Generated levels must keep at least one pixel along y and x."""

    @pytest.mark.parametrize(("factors", "shape"), [((), (1, 1, 1)), ((2, 2), (1, 4, 9)), ((4,), (2, 7, 4))])
    def test_accepts_factors_within_the_extent(self, factors: tuple[int, ...], shape: tuple[int, ...]) -> None:
        make_raster_options(factors).validate_image_pyramid(da.zeros(shape, chunks=shape), element="an image")

    def test_rejects_factors_beyond_the_smaller_extent_without_computing(self) -> None:
        data = da.from_delayed(_fail_if_computed(), shape=(2, 3, 9), dtype=np.uint8)

        with pytest.raises(ValueError, match=r"scale factors \(2, 2\) exceed the spatial extent of an image \(3 x 9"):
            make_raster_options((2, 2)).validate_image_pyramid(data, element="an image")


class TestValidateLabelPyramid:
    """Generated label levels must keep IDs exact and at least one pixel."""

    @pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.uint32, np.int32])
    def test_accepts_labels_of_at_most_32_bits(self, dtype: type[np.integer]) -> None:
        make_raster_options((2,)).validate_label_pyramid(da.zeros((4, 4), dtype=dtype), element="labels")

    @pytest.mark.parametrize("dtype", [np.uint64, np.int64])
    def test_rejects_wide_labels_only_with_generated_levels(self, dtype: type[np.integer]) -> None:
        labels = da.zeros((4, 4), dtype=dtype)
        make_raster_options().validate_label_pyramid(labels, element="labels")

        with pytest.raises(ValueError, match=rf"<=32-bit labels; cell labels has {np.dtype(dtype)} labels"):
            make_raster_options((2,)).validate_label_pyramid(labels, element="cell labels")

    def test_rejects_factors_beyond_the_extent(self) -> None:
        with pytest.raises(ValueError, match=r"exceed the spatial extent of labels \(1 x 8 pixels\)"):
            make_raster_options((2,)).validate_label_pyramid(da.zeros((1, 8), dtype=np.uint16), element="labels")


@delayed
def _fail_if_computed() -> np.ndarray:
    message = "pyramid validation computed pixels"
    raise AssertionError(message)
