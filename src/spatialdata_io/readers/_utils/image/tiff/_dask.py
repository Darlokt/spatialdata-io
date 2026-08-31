from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import tifffile
import zarr
from dask.base import tokenize

from spatialdata_io.readers._utils.errors import RasterFormatError
from spatialdata_io.readers._utils.image._types import LazyRaster
from spatialdata_io.readers._utils.image.tiff._types import (
    TiffMetadata,
    _TiffReadSpec,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

type ChunkSelection = tuple[slice, ...]


def _read_tiff_selection(spec: _TiffReadSpec, selection: ChunkSelection) -> NDArray[np.generic]:
    try:
        with tifffile.TiffFile(spec.path) as tiff:
            store = tiff.aszarr(series=spec.series, level=spec.level)
            try:
                source = zarr.open_array(store=store, mode="r")
                result = np.asarray(source[selection])
            finally:
                store.close()
    except Exception as error:
        error.add_note(
            f"Failed TIFF selection path={spec.path}, series={spec.series}, level={spec.level}, selection={selection}."
        )
        raise

    expected_shape = tuple(part.stop - part.start for part in selection)
    expected_dtype = np.dtype(spec.dtype)
    if result.shape != expected_shape or result.dtype != expected_dtype:
        message = (
            f"TIFF selection result does not match inspected metadata for {spec.path}: "
            f"series={spec.series}, level={spec.level}, selection={selection}, "
            f"expected shape={expected_shape}, dtype={expected_dtype}; "
            f"got shape={result.shape}, dtype={result.dtype}."
        )
        raise RasterFormatError(message)
    return result


@dataclass(frozen=True, slots=True)
class _TiffArray:
    """Path-backed array adapter used by Dask's compact blockwise layer."""

    spec: _TiffReadSpec
    shape: tuple[int, ...]
    dtype: np.dtype[np.generic]

    @property
    def ndim(self) -> int:
        """Number of stored array dimensions."""
        return len(self.shape)

    def __getitem__(self, selection: ChunkSelection) -> NDArray[np.generic]:
        selection_shape = tuple(
            (size if part.stop is None else part.stop) - (0 if part.start is None else part.start)
            for part, size in zip(selection, self.shape, strict=True)
        )
        if 0 in selection_shape:
            return np.empty(selection_shape, dtype=self.dtype)
        return _read_tiff_selection(self.spec, selection)


def read_tiff(
    metadata: TiffMetadata,
    *,
    series: int,
    level: int,
) -> LazyRaster:
    """Construct a native-chunked lazy TIFF array for an explicit series and level.

    The returned Dask graph stores paths and immutable scalar metadata only.
    Computing a slice executes only the TIFF-Zarr selections that intersect that
    slice; every task owns and closes its TIFF and Zarr resources.

    Parameters
    ----------
    metadata
        Closed immutable metadata returned by :func:`inspect_tiff`.
    series
        Explicit zero-based TIFF series index.
    level
        Explicit zero-based pyramid level within ``series``.

    Returns
    -------
    LazyRaster
        A Dask array aligned to native TIFF segments in the selected level's
        declared axis order.

    Raises
    ------
    IndexError
        If ``series`` or ``level`` is outside the inspected range.
    RasterFormatError
        If a task result disagrees with the inspected shape or dtype.

    Notes
    -----
    Rechunking the result is lazy. It does not change the native source layer:
    each source task still opens a short-lived ``tifffile`` Zarr store for one
    basic selection and closes it before returning.
    """
    if series < 0 or series >= len(metadata.series):
        message = f"TIFF series index {series} is outside [0, {len(metadata.series)})."
        raise IndexError(message)
    selected_series = metadata.series[series]
    if level < 0 or level >= len(selected_series.levels):
        message = f"TIFF level index {level} is outside [0, {len(selected_series.levels)}) for series {series}."
        raise IndexError(message)
    selected_level = selected_series.levels[level]
    spec = _TiffReadSpec(
        path=metadata.path,
        series=series,
        level=level,
        dtype=selected_level.dtype,
    )
    source = _TiffArray(spec=spec, shape=selected_level.shape, dtype=selected_level.dtype)
    name = f"spatialdata-io-tiff-{tokenize(source, selected_level.native_chunks)}"
    data = da.from_array(
        source,
        chunks=selected_level.native_chunks,
        name=name,
        lock=False,
        asarray=False,
        fancy=False,
    )
    return LazyRaster(data=data, axes=selected_level.axes)
