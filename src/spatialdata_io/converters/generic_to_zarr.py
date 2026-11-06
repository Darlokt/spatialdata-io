"""Persist generic SpatialData elements to Zarr stores."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from spatialdata import SpatialData

from spatialdata_io._raster_defaults import DEFAULT_SCALE_FACTORS
from spatialdata_io.readers.generic import generic

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spatialdata_io.readers.generic._images import Chunks

__all__ = ["generic_to_zarr"]


# Explicit public options mirror the reader and CLI without an opaque options mapping.
def generic_to_zarr(  # noqa: PLR0913, RUF100
    path: str | Path,
    output: str | Path,
    name: str | None = None,
    data_axes: str | None = None,
    coordinate_system: str | None = None,
    *,
    tiff_series: int | None = None,
    tiff_level: int = 0,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: Chunks | None = None,
    invalid_geometries: Literal["keep", "drop", "raise"] = "keep",
) -> None:
    """Read a generic element and persist it in a SpatialData Zarr store.

    Parameters
    ----------
    path
        Image or GeoJSON input supported by :func:`generic`.
    output
        Destination SpatialData Zarr store. A missing store is created and an
        existing store receives one new element.
    name
        Element name. Defaults to the input filename stem.
    data_axes
        Explicit image axes. Required for images and invalid for GeoJSON.
    coordinate_system
        Target frame name. ``None`` selects ``"global"`` for a new store.
        Explicit names must contain a non-whitespace character and are
        preserved literally. A name is required when appending: it asserts
        the intended correspondence to an existing frame, or names a separate frame.
        Calibrated images use micrometres; GeoJSON retains source units.
        Naming a shared frame does not convert or register GeoJSON coordinates.
    tiff_series
        Zero-based TIFF series. ``None`` reads the only series and fails for a
        TIFF with several. Must be ``None`` for non-TIFF inputs.
    tiff_level
        Zero-based TIFF pyramid level. Must remain zero for non-TIFF inputs.
    scale_factors
        Integer factors of at least two, one per generated image pyramid
        level; empty for a single scale. Invalid for GeoJSON inputs.
    chunks
        One positive image chunk size per canonical axis, applied at every
        pyramid level: an integer, a tuple in canonical ``CYX``/``CZYX`` order,
        or a mapping keyed by exactly the canonical axes. ``None`` selects
        ``(1, 1024, 1024)`` for ``CYX`` and ``(1, 1, 1024, 1024)`` for
        ``CZYX``. Invalid for GeoJSON inputs.
    invalid_geometries
        GeoJSON topology policy, forwarded to :func:`generic`. Defaults to
        preserving source polygons; malformed geometry fails in every mode.

    Raises
    ------
    TypeError
        If a frame name is not a string, or image scale factors or chunk sizes
        are not integers.
    ValueError
        If the input options are invalid or ``name`` already exists.

    Notes
    -----
    The function writes no status text to standard output. CLI presentation is
    owned by the command wrapper.
    """
    if coordinate_system is not None:
        if not isinstance(coordinate_system, str):
            message = "coordinate_system must be a string or None."
            raise TypeError(message)
        if not coordinate_system.strip():
            message = "coordinate_system must contain a non-whitespace character."
            raise ValueError(message)

    input_path = Path(path)
    output_path = Path(output)
    element_name = input_path.stem if name is None else name

    spatial_data: SpatialData | None = None
    if output_path.exists():
        spatial_data = SpatialData.read(output_path)
        if element_name in spatial_data:
            message = (
                f"Name {element_name!r} already exists in {output_path}; provide a different "
                "name or delete the existing element."
            )
            raise ValueError(message)
        if coordinate_system is None:
            message = "Appending requires an explicit coordinate_system to establish a shared or independent frame."
            raise ValueError(message)

    element = generic(
        path=input_path,
        data_axes=data_axes,
        coordinate_system=coordinate_system,
        tiff_series=tiff_series,
        tiff_level=tiff_level,
        scale_factors=scale_factors,
        chunks=chunks,
        invalid_geometries=invalid_geometries,
    )

    if spatial_data is not None:
        spatial_data[element_name] = element
        spatial_data.write_element(element_name=element_name)
        return

    spatial_data = SpatialData.init_from_elements(elements={element_name: element})
    spatial_data.write(output_path)
