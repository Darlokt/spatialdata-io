"""Central pyramid and chunk options shared by every reader that returns images or labels.

Images default to one channel per chunk and square 1024-pixel y/x tiles, labels
to the same y/x tiles, and no generated pyramid levels. The values live in the
dependency-free ``spatialdata_io._raster_defaults``, shared with the
command-line interface; readers use them as visible signature defaults, so one
edit there changes every reader and command. Model chunks are always passed
explicitly: without them the multiscale image builder rechunks to its own
256-pixel tiles and merges all channels into one chunk.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS, DEFAULT_TILE_SIZE

__all__ = ["DEFAULT_CHUNKS", "DEFAULT_SCALE_FACTORS", "DEFAULT_TILE_SIZE", "RasterOptions", "make_raster_options"]

_IMAGE_AXES = ("c", "y", "x")
_MIN_SCALE_FACTOR = 2


@dataclass(frozen=True, slots=True)
class RasterOptions:
    """Validated pyramid factors and CYX chunks; labels use the y/x sizes.

    ``scale_factors`` is empty for a single scale.
    """

    scale_factors: tuple[int, ...]
    chunks: int | tuple[int, int, int]

    @property
    def model_scale_factors(self) -> list[int] | None:
        """Return the factors in the form SpatialData models expect."""
        return list(self.scale_factors) if self.scale_factors else None

    def image_chunks(self) -> dict[str, int]:
        """Return explicit ``c, y, x`` chunk sizes of an image element."""
        if isinstance(self.chunks, int):
            return dict.fromkeys(_IMAGE_AXES, self.chunks)
        return dict(zip(_IMAGE_AXES, self.chunks, strict=True))

    def label_chunks(self) -> dict[str, int]:
        """Return explicit ``y, x`` chunk sizes of a labels element."""
        sizes = self.image_chunks()
        return {"y": sizes["y"], "x": sizes["x"]}


def make_raster_options(
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
) -> RasterOptions:
    """Validate public pyramid and chunk options once and own the caller's sequence.

    Parameters
    ----------
    scale_factors
        Integer downsampling factors of at least two, one per generated pyramid
        level, applied equally to images and labels. Empty for a single scale.
    chunks
        One positive size for every axis, or three positive sizes in ``c, y, x``
        order. Labels use the ``y`` and ``x`` sizes.

    Raises
    ------
    TypeError
        If the factors are not a sequence of integers, or chunks are neither an
        integer nor a tuple of integers.
    ValueError
        If a factor is below two, or chunks are not positive or a tuple does not
        have three sizes.
    """
    if isinstance(scale_factors, str | bytes) or not isinstance(scale_factors, Sequence):
        message = "scale_factors must be a sequence of integer downsampling factors."
        raise TypeError(message)
    factors = tuple(scale_factors)
    if any(isinstance(factor, bool) or not isinstance(factor, int) for factor in factors):
        message = "scale_factors must contain integers."
        raise TypeError(message)
    if any(factor < _MIN_SCALE_FACTOR for factor in factors):
        message = "scale_factors must contain factors of at least 2; pass () for a single scale."
        raise ValueError(message)

    # A list, None or a nested value is not an integer size and fails here.
    sizes = chunks if isinstance(chunks, tuple) else (chunks,)
    integers = [size for size in sizes if isinstance(size, int) and not isinstance(size, bool)]
    if len(integers) != len(sizes):
        message = "chunks must be a positive integer or a tuple of three positive c, y, x integers."
        raise TypeError(message)
    if (isinstance(chunks, tuple) and len(integers) != len(_IMAGE_AXES)) or min(integers) <= 0:
        message = "chunks must be positive, with exactly three c, y, x sizes when a tuple is given."
        raise ValueError(message)
    return RasterOptions(factors, chunks)
