"""Persist generic SpatialData elements to Zarr stores."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from spatialdata import SpatialData
from spatialdata._docs import docstring_parameter

from spatialdata_io.readers.generic import (
    VALID_IMAGE_TYPES,
    VALID_SHAPE_TYPES,
    generic,
)

if TYPE_CHECKING:
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
    chunks
        Requested image chunks in canonical output-axis order or by canonical
        axis name. Invalid for GeoJSON inputs. By default, only ``Y`` and ``X``
        are rechunked to 1024; ``C`` and ``Z`` retain source chunking.

    Raises
    ------
    TypeError
        If an image chunk specification contains unsupported value types.
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
        chunks=chunks,
    )

    if spatial_data is not None:
        spatial_data[element_name] = element
        spatial_data.write_element(element_name=element_name)
        return

    spatial_data = SpatialData.init_from_elements(elements={element_name: element})
    spatial_data.write(output_path)
