"""Normalize declared OME pixel lengths without assigning coordinate meaning."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, Literal

from spatialdata_io.readers._utils.errors import ReaderFormatError

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.image.tiff import TiffMetadata

_MICROMETERS_PER_UNIT = {"µm": 1.0, "nm": 0.001, "mm": 1000.0}


def ome_pixel_sizes_um(
    metadata: TiffMetadata,
    *,
    series: int,
    axes: tuple[Literal["x", "y", "z"], ...],
    context: ReaderErrorContext,
) -> tuple[float | None, ...]:
    """Return each requested declared OME pixel length in micrometres.

    Parameters
    ----------
    metadata
        Borrowed, read-only metadata of one inspected TIFF.
    series
        Selected TIFF series; inspection guarantees one OME image per series.
    axes
        Spatial axes whose declarations are requested, in output order.
    context
        Owning reader and artifact for malformed declaration errors.

    Returns
    -------
    tuple of float or None
        One entry per requested axis: a finite positive length in micrometres,
        or ``None`` when that size is not declared. Every entry is ``None``
        without OME metadata.

    Raises
    ------
    ReaderFormatError
        A present size or its unit violates the supported length contract,
        even when other requested sizes are absent.

    Notes
    -----
    Sizes in µm, nm and mm are supported. The reader decides whether a partial
    declaration calibrates anything and owns axis association, dimensions,
    agreement between artifacts, registration and transformation assembly.
    No source metadata is changed and no pixel data is read.
    """
    if metadata.ome is None:
        return (None,) * len(axes)
    pixels = metadata.ome.images[series].pixels
    fields = {
        "x": (pixels.physical_size_x, pixels.physical_size_x_unit),
        "y": (pixels.physical_size_y, pixels.physical_size_y_unit),
        "z": (pixels.physical_size_z, pixels.physical_size_z_unit),
    }
    result: list[float | None] = []
    for axis in axes:
        value, unit = fields[axis]
        if value is None:
            result.append(None)
            continue
        factor = _MICROMETERS_PER_UNIT.get(unit.value)
        if factor is None or not isfinite(value) or value <= 0 or not isfinite(value * factor) or value * factor <= 0:
            raise ReaderFormatError(
                context=context,
                path=metadata.path,
                found=f"PhysicalSize{axis.upper()}={value!r} {unit.value}",
                expected="finite positive physical sizes in µm, nm or mm, representable in µm",
            )
        result.append(value * factor)
    return tuple(result)
