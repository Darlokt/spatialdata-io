"""Interpret seqFISH OME pixel calibration in the ROI's micrometer frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata.transformations import Scale

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import ome_pixel_sizes_um

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.image import TiffMetadata


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

    y, x = ome_pixel_sizes_um(metadata, series=0, axes=("y", "x"), context=context)
    if y is None or x is None:
        raise ReaderFormatError(
            context=context,
            path=metadata.path,
            found=f"physical Y/X sizes {(y, x)!r}",
            expected="finite positive physical X/Y sizes in µm, nm or mm",
        )
    return Scale([y, x], axes=("y", "x"))
