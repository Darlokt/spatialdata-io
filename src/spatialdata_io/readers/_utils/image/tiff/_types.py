from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from ome_types.model import OME


@dataclass(frozen=True, slots=True)
class TiffLevelMetadata:
    """Storage facts for one TIFF series level.

    ``axes`` and ``shape`` describe the array exposed by ``tifffile`` for this
    level. ``native_chunks`` is the exact selection geometry exposed by its Zarr
    store. No semantic axis selection or pyramid interpretation is applied.
    """

    shape: tuple[int, ...]
    dtype: np.dtype[np.generic]
    axes: tuple[str, ...]
    native_chunks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TiffSeriesMetadata:
    """Ordered storage levels for one TIFF image series."""

    levels: tuple[TiffLevelMetadata, ...]


@dataclass(frozen=True, slots=True)
class TiffMetadata:
    """Closed metadata for one logical TIFF source.

    ``ome`` is the complete parsed OME model when OME-XML is present. It exposes
    logical XYZCT sizes, physical sizes and units, channels, planes, acquisition
    metadata, and structured annotations without duplicating a selected subset
    in this package. Callers must treat the owned model as read-only. No open
    TIFF files or Zarr stores escape inspection.
    """

    path: Path
    series: tuple[TiffSeriesMetadata, ...]
    ome: OME | None


@dataclass(frozen=True, slots=True)
class _TiffReadSpec:
    """Minimal immutable state embedded in TIFF pixel tasks."""

    path: Path
    series: int
    level: int
    dtype: np.dtype[np.generic]
