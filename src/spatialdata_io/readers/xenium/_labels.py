"""Build lazy segmentation labels from path-backed ``cells.zarr.zip`` mask chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import zarr
from dask.base import tokenize
from spatialdata.models import Labels2DModel
from spatialdata.transformations import Scale
from zarr.storage import ZipStore

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers._utils.paths import ArtifactFile


@dataclass(frozen=True, slots=True)
class MaskSource:
    """Picklable description of one mask array; every read reopens the Zip store.

    Graphs therefore hold a path and immutable metadata, never an open archive.
    """

    store: ArtifactFile
    key: str
    shape: tuple[int, int]
    chunks: tuple[int, int]

    @property
    def ndim(self) -> int:
        """Return the array rank required by Dask."""
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype[np.uint32]:
        """Return the element type required by Dask; layout discovery admits only uint32 masks."""
        return np.dtype(np.uint32)

    def __getitem__(self, selection: tuple[slice, slice]) -> NDArray[np.uint32]:
        """Read one bounded selection and close the store before returning."""
        with ZipStore(self.store, read_only=True) as store:
            array = zarr.open_array(store=store, path=self.key, mode="r", zarr_format=2)
            return np.asarray(array[selection])


def read_labels(source: MaskSource, *, pixel_size: float, raster: RasterOptions) -> DataArray | DataTree:
    """Return one mask as a lazy ``(y, x)`` labels element in the micrometre frame.

    Tasks follow the native Zarr chunks, about one eighth of the image per axis in
    vendor stores, so even a small selection decodes whole vendor chunks.
    """
    data = da.from_array(
        source,
        chunks=source.chunks,
        name=f"xenium-mask-{tokenize(source)}",
        asarray=False,
        fancy=False,
        meta=np.empty((0, 0), dtype=source.dtype),
    )
    return Labels2DModel.parse(
        data,
        dims=("y", "x"),
        transformations={"global": Scale([pixel_size, pixel_size], axes=("y", "x"))},
        scale_factors=raster.model_scale_factors,
        chunks=raster.label_chunks(),
    )
