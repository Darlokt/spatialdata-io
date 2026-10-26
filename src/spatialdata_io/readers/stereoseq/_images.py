"""Build native-chunk registered-image graphs through the shared TIFF boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import inspect_tiff, normalize_raster_axes, read_tiff

if TYPE_CHECKING:
    from xarray import DataArray, DataTree

    from spatialdata_io.readers.stereoseq._layout import ImageSource


def read_image(source: ImageSource, *, chunks: int | tuple[int, ...] | None) -> DataArray | DataTree:
    """Read one full-resolution 2D image lazily; masks retain image semantics."""
    metadata = inspect_tiff(source.file)
    if len(metadata.series) != 1:
        raise ReaderFormatError(
            context=ReaderErrorContext("stereoseq", source.name),
            path=source.file,
            found=f"{len(metadata.series)} TIFF series",
            expected="one registered image series",
        )

    raster = read_tiff(metadata, series=0, level=0)
    # TIFF samples carry RGB(A) semantics; scientific C axes retain channel order.
    axes = tuple("c" if axis.lower() == "s" else axis.lower() for axis in raster.axes)
    data = normalize_raster_axes(raster.data, axes, target_axes=("c", "y", "x"), add_missing=("c",))
    if chunks is not None:
        data = data.rechunk(chunks)

    channel_count = int(data.shape[0])
    rgb_samples = "s" in tuple(axis.lower() for axis in raster.axes) and channel_count in (3, 4)
    channels = ["r", "g", "b", "a"][:channel_count] if rgb_samples else None
    return Image2DModel.parse(
        data, dims=("c", "y", "x"), c_coords=channels, rgb=False, transformations={"global": Identity()}
    )
