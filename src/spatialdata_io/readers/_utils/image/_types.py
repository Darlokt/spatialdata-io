from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

    import dask.array as da
    import numpy as np


@dataclass(frozen=True, slots=True)
class LazyRaster:
    """A lazy raster array in decoder-declared axis order.

    Attributes
    ----------
    data
        Dask array whose graph owns no open source resources.
    axes
        Single-character axis names in the current array order.
    """

    data: da.Array
    axes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _EncodedHeader:
    """Private decoder output facts established by bounded inspection."""

    shape: tuple[int, ...]
    dtype: np.dtype[np.generic]
    axes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _EncodedReadSpec:
    """Minimal immutable state embedded in an encoded-image task."""

    path: Path
    codec: Literal["png", "jpeg"]
    shape: tuple[int, ...]
    dtype: np.dtype[np.generic]
