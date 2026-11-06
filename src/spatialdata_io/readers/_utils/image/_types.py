"""Lazy raster result, declared sample colours and the private state of encoded-image tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import dask.array as da
    import numpy as np

    from spatialdata_io.readers._utils.paths import ArtifactFile

# The colour model a source declares for its interleaved samples axis: red,
# green and blue samples, optionally followed by one alpha sample. Each letter
# names one sample in order.
type SampleColor = Literal["rgb", "rgba"]


@dataclass(frozen=True, slots=True)
class LazyRaster:
    """A lazy raster array in decoder-declared axis order.

    Attributes
    ----------
    data
        Dask array whose graph owns no open source resources.
    axes
        Single-character axis names in the current array order. ``s`` is the
        source's interleaved samples axis (TIFF ``S``, PNG and JPEG samples).
    color
        The colour model the source declares for its samples axis ``s``, or
        ``None`` when it declares none (grayscale, other photometric models,
        unspecified extra samples, or no samples axis).
    """

    data: da.Array
    axes: tuple[str, ...]
    color: SampleColor | None = None


@dataclass(frozen=True, slots=True)
class _EncodedHeader:
    """Private decoder output facts established by bounded inspection."""

    shape: tuple[int, ...]
    dtype: np.dtype[np.generic]
    axes: tuple[str, ...]
    color: SampleColor | None


@dataclass(frozen=True, slots=True)
class _EncodedReadSpec:
    """Minimal immutable state embedded in an encoded-image task."""

    path: ArtifactFile
    codec: Literal["png", "jpeg"]
    shape: tuple[int, ...]
    dtype: np.dtype[np.generic]
