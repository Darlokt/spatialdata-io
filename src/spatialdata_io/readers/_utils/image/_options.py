"""Central pyramid and chunk options shared by every reader that returns images or labels.

Images default to one channel per chunk and square 1024-pixel y/x tiles, labels
to the same y/x tiles, and no generated pyramid levels. The values live in the
dependency-free ``spatialdata_io._raster_defaults``, shared with the
command-line interface; readers use them as visible signature defaults, so one
edit there changes every reader and command. Model chunks are always passed
explicitly: without them the multiscale image builder rechunks to its own
256-pixel tiles and merges all channels into one chunk. Two guards reject
generated levels that would shrink an element below one pixel or change label
IDs wider than 32 bits.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS

if TYPE_CHECKING:
    import dask.array as da

__all__ = ["RasterOptions", "make_raster_options"]

_IMAGE_AXES = ("c", "y", "x")
_MIN_SCALE_FACTOR = 2
# Generated label levels keep IDs of at most 32 bits exactly; SpatialData's
# downsampling turns a uint64 ID of 2**53 + 1 into 2**53.
_MAX_EXACT_LABEL_BYTES = 4


@dataclass(frozen=True, slots=True)
class RasterOptions:
    """Validated pyramid factors and ``c, y, x`` chunks; labels use the y/x sizes.

    ``scale_factors`` is empty for a single scale.
    """

    scale_factors: tuple[int, ...]
    chunks: tuple[int, int, int]

    @property
    def model_scale_factors(self) -> list[int] | None:
        """Return the factors in the form SpatialData models expect."""
        return list(self.scale_factors) if self.scale_factors else None

    def image_chunks(self) -> dict[str, int]:
        """Return explicit ``c, y, x`` chunk sizes of an image element."""
        return dict(zip(_IMAGE_AXES, self.chunks, strict=True))

    def label_chunks(self) -> dict[str, int]:
        """Return explicit ``y, x`` chunk sizes of a labels element."""
        _, y, x = self.chunks
        return {"y": y, "x": x}

    def validate_image_pyramid(self, data: da.Array, /, *, element: str) -> None:
        """Reject scale factors that would reduce an image below one pixel.

        Parameters
        ----------
        data
            Lazy image whose last two axes are ``y`` and ``x``. Only its shape
            is inspected.
        element
            Reader description of the element for the error message.

        Raises
        ------
        ValueError
            If the product of the factors exceeds the smaller y/x extent.
        """
        extent = data.shape[-2:]
        if self.scale_factors and min(extent) < prod(self.scale_factors):
            message = (
                f"Requested scale factors {self.scale_factors} exceed the spatial extent of {element} "
                f"({extent[0]} x {extent[1]} pixels)."
            )
            raise ValueError(message)

    def validate_label_pyramid(self, data: da.Array, /, *, element: str) -> None:
        """Reject generated label levels that would change IDs or vanish.

        Parameters
        ----------
        data
            Lazy labels whose last two axes are ``y`` and ``x``. Only the shape
            and dtype are inspected.
        element
            Reader description of the element for error messages.

        Raises
        ------
        ValueError
            If labels wider than 32 bits would be downsampled, which rounds
            IDs above 2**53, or the factors exceed the spatial extent.
        """
        if self.scale_factors and data.dtype.itemsize > _MAX_EXACT_LABEL_BYTES:
            message = (
                f"Generated label pyramids require <=32-bit labels; {element} has {data.dtype} labels. "
                "Use scale_factors=() to keep 64-bit IDs exact."
            )
            raise ValueError(message)
        self.validate_image_pyramid(data, element=element)


def make_raster_options(
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
) -> RasterOptions:
    """Validate public pyramid and chunk options once and own the caller's sequence.

    An integer ``chunks`` value is expanded to the same size for ``c``, ``y``
    and ``x``, so the result has one representation.

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
    return RasterOptions(factors, chunks if isinstance(chunks, tuple) else (chunks, chunks, chunks))
