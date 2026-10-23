"""Interpret seqFISH OME pixel calibration in the ROI's micrometer frame."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

from spatialdata.transformations import Scale

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.image import TiffMetadata

_MICROMETERS_PER_UNIT = {"µm": 1.0, "nm": 0.001, "mm": 1000.0}


def pixel_scale(metadata: TiffMetadata) -> Scale:
    """Return explicit Y/X pixel sizes converted to micrometers.

    OME defaults omitted physical-size units to micrometers. Only micrometer,
    nanometer and millimeter calibrations are supported; missing, nonfinite or
    nonpositive sizes cannot define the physical ROI coordinate system.
    """
    context = ReaderErrorContext("seqfish", "DAPI OME calibration")
    ome = metadata.ome
    if ome is None or len(ome.images) != 1:
        raise ReaderFormatError(
            context=context,
            path=metadata.path,
            found="missing OME metadata or an ambiguous OME image collection",
            expected="one OME image with physical X/Y pixel sizes",
        )

    pixels = ome.images[0].pixels
    sizes: list[float] = []
    for axis, value, unit in (
        ("Y", pixels.physical_size_y, pixels.physical_size_y_unit),
        ("X", pixels.physical_size_x, pixels.physical_size_x_unit),
    ):
        factor = _MICROMETERS_PER_UNIT.get(unit.value)
        if value is None or not isfinite(value) or value <= 0 or factor is None:
            raise ReaderFormatError(
                context=context,
                path=metadata.path,
                found=f"PhysicalSize{axis}={value!r} {unit.value}",
                expected="finite positive physical X/Y sizes in µm, nm or mm",
            )
        converted = value * factor
        if not isfinite(converted) or converted <= 0:
            raise ReaderFormatError(
                context=context,
                path=metadata.path,
                found=f"PhysicalSize{axis} overflows or underflows after conversion to µm",
                expected="finite positive pixel sizes in µm",
            )
        sizes.append(converted)

    return Scale(sizes, axes=("y", "x"))
