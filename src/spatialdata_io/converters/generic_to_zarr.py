"""Persist generic SpatialData elements to Zarr stores."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from spatialdata import SpatialData
from spatialdata._docs import docstring_parameter

from spatialdata_io.readers._utils.image import DEFAULT_SCALE_FACTORS
from spatialdata_io.readers.generic import (
    VALID_IMAGE_TYPES,
    VALID_SHAPE_TYPES,
    generic,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spatialdata_io.readers.generic._images import Chunks

__all__ = ["generic_to_zarr"]


@docstring_parameter(
    valid_image_types=", ".join(VALID_IMAGE_TYPES),
    valid_shape_types=", ".join(VALID_SHAPE_TYPES),
)
# Explicit public options mirror the reader and CLI without an opaque options mapping.
def generic_to_zarr(  # noqa: PLR0913, RUF100
    path: str | Path,
    output: str | Path,
    name: str | None = None,
    data_axes: str | None = None,
    coordinate_system: str | None = None,
    *,
    tiff_series: int = 0,
    tiff_level: int = 0,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: Chunks | None = None,
) -> None:
    """Read a generic element and persist it in a SpatialData Zarr store.

    Parameters
    ----------
    path
        Image or GeoJSON input. Supported image extensions are
        {valid_image_types}; supported shape extensions are {valid_shape_types}.
    output
        Destination SpatialData Zarr store. A missing store is created and an
        existing store receives one new element.
    name
        Element name. Defaults to the input filename stem.
    data_axes
        Explicit image axes. Required for images and invalid for GeoJSON.
    coordinate_system
        Coordinate system for the element. Defaults to ``"global"``.
    tiff_series
        Zero-based TIFF series. Must remain zero for non-TIFF inputs.
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

    Raises
    ------
    TypeError
        If image scale factors or chunk sizes are not integers.
    ValueError
        If the input options are invalid or ``name`` already exists.

    Notes
    -----
    The function writes no status text to standard output. CLI presentation is
    owned by the command wrapper.
    """
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

    element = generic(
        path=input_path,
        data_axes=data_axes,
        coordinate_system=coordinate_system,
        tiff_series=tiff_series,
        tiff_level=tiff_level,
        scale_factors=scale_factors,
        chunks=chunks,
    )

    if spatial_data is not None:
        spatial_data[element_name] = element
        spatial_data.write_element(element_name=element_name)
        return

    spatial_data = SpatialData.init_from_elements(elements={element_name: element})
    spatial_data.write(output_path)
