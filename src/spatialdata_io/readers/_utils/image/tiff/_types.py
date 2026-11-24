"""Describe the TIFF storage available to path-backed raster tasks.

A series level describes the logical array exposed by tifffile. A physical
page describes one top-level image file directory, which can be a thumbnail
or an individual measurement inside a series. Both record shape, dtype,
axes and native chunk geometry. Optional page metadata preserves file order
and the original rational resolution and position tags so a format reader
can interpret their units. These records outlive the inspection file handle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from ome_types.model import OME

    from spatialdata_io.readers._utils.image._types import SampleColor
    from spatialdata_io.readers._utils.paths import ArtifactFile


@dataclass(frozen=True, slots=True)
class _TiffRasterMetadata:
    """Storage facts of one TIFF series level or physical page.

    ``axes`` and ``shape`` describe the array exposed by ``tifffile`` for this
    level. ``native_chunks`` is the exact selection geometry exposed by its Zarr
    store. ``color`` is the colour model the level declares for its samples
    axis ``S`` (``None`` without one). No semantic axis selection or pyramid
    interpretation is applied.
    """

    shape: tuple[int, ...]
    dtype: np.dtype[np.generic]
    axes: tuple[str, ...]
    native_chunks: tuple[int, ...]
    color: SampleColor | None
    direct_row_axis: int | None = field(default=None, kw_only=True)


@dataclass(frozen=True, slots=True)
class TiffLevelMetadata(_TiffRasterMetadata):
    """Storage geometry of a series level in its declared logical dimensions."""


@dataclass(frozen=True, slots=True)
class TiffPageMetadata(_TiffRasterMetadata):
    """Closed storage geometry and selected standard tags from one top-level IFD.

    Empty description or software strings indicate absent or empty text tags.
    Optional numeric tags preserve absence and source values, including exact
    rational components. Min/max sample values are tags 340 and 341.
    """

    index: int
    description: str
    software: str
    x_resolution: tuple[int, int] | None
    y_resolution: tuple[int, int] | None
    resolution_unit: int | None
    x_position: tuple[int, int] | None
    y_position: tuple[int, int] | None
    orientation: int | None
    photometric: int
    samples_per_pixel: int
    min_sample_value: int | float | tuple[int | float, ...] | None
    max_sample_value: int | float | tuple[int | float, ...] | None


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

    path: ArtifactFile
    series: tuple[TiffSeriesMetadata, ...]
    ome: OME | None
    pages: tuple[TiffPageMetadata, ...] | None = None
    imagej_labels: tuple[str, ...] | None = None
