"""Read Xenium morphology images lazily with their OME channel names.

XOA 1.x writes single-file ``morphology_focus.ome.tif`` and
``morphology_mip.ome.tif``. XOA 2.0+ writes one OME-TIFF per channel in
``morphology_focus/`` (``morphology_focus_NNNN`` or, from 4.0,
``chNNNN_<marker>``); every file lists the complete UUID-linked set, whose
channel order is the OME channel order. File names are not channel labels.
Protein channels carry a documented +100 intensity offset in their metadata;
pixels are returned as recorded.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ome_types.model import UnitsLength
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import inspect_tiff, normalize_raster_axes, read_tiff

if TYPE_CHECKING:
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers.xenium._layout import FocusImages

# Pixel sizes are decimal text in both experiment.xenium and the OME metadata.
_PIXEL_SIZE_RTOL = 1e-9


def read_morphology(images: FocusImages, *, pixel_size: float, raster: RasterOptions) -> DataArray | DataTree:
    """Return one CYX morphology image in the micrometre frame without reading pixels.

    Raises
    ------
    ReaderFormatError
        If the file has not exactly one OME image series, has unnamed or repeated
        channels, a calibration other than ``pixel_size``, or a linked file set
        different from the OME-TIFF files beside it.
    """
    context = ReaderErrorContext("xenium", images.entry.name)
    try:
        metadata = inspect_tiff(images.entry)
    except (OSError, ValueError) as caught:
        add_reader_context(caught, context=context, path=images.entry)
        raise

    def error(*, found: str, expected: str) -> ReaderFormatError:
        return ReaderFormatError(context=context, path=images.entry, found=found, expected=expected)

    if len(metadata.series) != 1 or metadata.ome is None or len(metadata.ome.images) != 1:
        raise error(found=f"{len(metadata.series)} series", expected="one OME image series")
    pixels = metadata.ome.images[0].pixels

    names = [channel.name for channel in pixels.channels]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise error(found=f"channel names {names!r}", expected="unique nonempty OME channel names")
    for size, unit in (
        (pixels.physical_size_x, pixels.physical_size_x_unit),
        (pixels.physical_size_y, pixels.physical_size_y_unit),
    ):
        if (
            size is None
            or unit != UnitsLength.MICROMETER
            or not math.isclose(size, pixel_size, rel_tol=_PIXEL_SIZE_RTOL)
        ):
            raise error(found=f"physical pixel size {size} {unit}", expected=f"{pixel_size} µm from experiment.xenium")

    # A block without a file reference lives in the entry file itself.
    linked = {
        images.entry
        if block.uuid is None or block.uuid.file_name is None
        else images.entry.parent / block.uuid.file_name
        for block in pixels.tiff_data_blocks
    }
    if {path.resolve() for path in linked} != set(images.files):
        raise error(
            found=f"OME-linked files {sorted(path.name for path in linked)}",
            expected=f"exactly the OME-TIFF files {sorted(path.name for path in images.files)}",
        )

    lazy = read_tiff(metadata, series=0, level=0)
    data = normalize_raster_axes(lazy.data, lazy.axes, target_axes=("c", "y", "x"), add_missing=("c",))
    if data.shape[0] != len(names):
        raise error(found=f"{data.shape[0]} channels for {len(names)} names", expected="one OME name per channel")
    return Image2DModel.parse(
        data,
        dims=("c", "y", "x"),
        c_coords=[str(name) for name in names],
        rgb=False,
        scale_factors=raster.model_scale_factors,
        chunks=raster.image_chunks(),
        transformations={"global": Scale([pixel_size, pixel_size], axes=("y", "x"))},
    )
