"""Shared pyramid and chunk options with central defaults."""

from __future__ import annotations

import pytest

from spatialdata_io.readers._utils.image import (
    DEFAULT_CHUNKS,
    DEFAULT_SCALE_FACTORS,
    DEFAULT_TILE_SIZE,
    make_raster_options,
)


class TestMakeRasterOptions:
    """Options validate once, own the caller's factors and give explicit model chunks."""

    def test_defaults(self) -> None:
        options = make_raster_options()

        assert DEFAULT_TILE_SIZE == 1024
        assert DEFAULT_CHUNKS == (1, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE)
        assert DEFAULT_SCALE_FACTORS == ()
        assert options.scale_factors == ()
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
        assert make_raster_options((), 64).image_chunks() == {"c": 64, "y": 64, "x": 64}

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
