"""Map seqFISH DAPI pixels into the ROI's micrometre frame from OME physical sizes.

``<ROI>_DAPI.tiff`` is an OME-TIFF whose single image declares
``PhysicalSizeX`` and ``PhysicalSizeY`` (OME defaults an omitted unit to µm).
The pixel-to-µm transformation is ``Scale([PhysicalSizeY, PhysicalSizeX])``
on ``(y, x)``; anisotropic pixels keep their two sizes. The same scale maps the
segmentation and the boundary polygons, which share the DAPI pixel grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata.transformations import Scale

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image import ome_pixel_sizes_um

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.image import TiffMetadata


def pixel_scale(metadata: TiffMetadata, *, context: ReaderErrorContext) -> Scale:
    """Return explicit Y/X pixel sizes converted to micrometres.

    Only micrometre, nanometre and millimetre calibrations are supported;
    missing, nonfinite or nonpositive sizes cannot define the physical ROI
    coordinate system.
    """
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
