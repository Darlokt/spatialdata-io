"""Construct storage-aligned TIFF chunk graphs with task-owned file and store lifetimes."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import tifffile
import zarr
from dask.base import tokenize

from spatialdata_io.readers._utils.errors import RasterFormatError
from spatialdata_io.readers._utils.image._types import LazyRaster
from spatialdata_io.readers._utils.image.tiff._types import (
    TiffLevelMetadata,
    TiffMetadata,
    _TiffReadSpec,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

type ChunkSelection = tuple[slice, ...]

# Upper bound for the decoded bytes of one task that groups whole-row chunks.
# Every task opens its TIFF, so one task per small strip (often a single row)
# dominates read time. Measured on one-row-strip exports, 16 MiB bands match
# whole-plane conversion speed while keeping partial reads bounded.
STRIP_BAND_BYTES = 16 * 2**20


def _task_chunks(level: TiffLevelMetadata) -> tuple[int, ...]:
    """Return the chunk geometry of one source task for a TIFF level.

    Levels whose native chunks span every axis after ``Y`` store whole rows:
    strips, or tiles as wide as the image. Consecutive chunks of such levels
    are grouped into row bands of at most ``STRIP_BAND_BYTES`` decoded bytes
    that start at chunk boundaries; a chunk larger than a band stays alone.
    Other tiled levels keep their native chunks.
    """
    chunks = level.native_chunks
    if "Y" not in level.axes:
        return chunks

    row_axis = level.axes.index("Y")
    trailing = range(row_axis + 1, len(chunks))
    if row_axis == len(chunks) - 1 or any(chunks[axis] != level.shape[axis] for axis in trailing):
        return chunks

    chunk_bytes = prod(chunks) * level.dtype.itemsize
    rows = min(level.shape[row_axis], chunks[row_axis] * max(1, STRIP_BAND_BYTES // chunk_bytes))
    return (*chunks[:row_axis], rows, *chunks[row_axis + 1 :])


def _read_contiguous_rows(
    tiff: tifffile.TiffFile, spec: _TiffReadSpec, selection: ChunkSelection
) -> NDArray[np.generic] | None:
    """Read whole rows of one uncompressed contiguous page with a single bounded read.

    tifffile's Zarr store decodes one strip per chunk, and Zarr's per-chunk
    overhead is about ten times the cost of reading a one-row strip. A page
    that tifffile marks final (uncompressed and contiguous, no predictor,
    default fill order, no subsampling) stores the array exactly as laid out in
    memory, so whole rows are copied directly, as ``TiffPage.asarray`` does for
    whole pages. Returns None whenever the general path is required.
    """
    level = tiff.series[spec.series].levels[spec.level]
    page = level.keyframe
    stored = level.pages[0] if len(level.pages) == 1 else None
    if stored is None or page.dtype is None or not page.is_final:
        return None

    # The page must be stored in this file (OME series can link other files),
    # and its segments must hold exactly the page: tifffile also reports a
    # single empty or truncated strip as contiguous.
    stored_exactly = stored.parent is tiff and stored.dataoffsets[0] > 0 and sum(stored.databytecounts) == page.nbytes
    separate_samples, depth, _, width, samples = page.shaped
    # A series transform (such as MD Gel scaling) changes values and dtype.
    plain_layout = (
        (separate_samples, depth) == (1, 1)
        and level.transform is None
        and level.dtype == page.dtype
        and tuple(level.shape) == page.shape
    )
    whole_rows = all(
        (part.start, part.stop) == (0, size) for part, size in zip(selection[1:], page.shape[1:], strict=True)
    )
    if not (stored_exactly and plain_layout and whole_rows):
        return None

    rows = selection[0]
    row_items = width * samples
    # Interpret the bytes in file byte order; read_array returns native order,
    # which matches the dtype recorded at inspection.
    dtype = np.dtype(tiff.byteorder + page.dtype.char)
    tiff.filehandle.seek(stored.dataoffsets[0] + rows.start * row_items * dtype.itemsize)
    data = tiff.filehandle.read_array(dtype, count=(rows.stop - rows.start) * row_items)
    return data.reshape(rows.stop - rows.start, *page.shape[1:])


def _read_tiff_selection(spec: _TiffReadSpec, selection: ChunkSelection) -> NDArray[np.generic]:
    """Decode one bounded selection and close every task-owned resource."""
    try:
        with tifffile.TiffFile(spec.path) as tiff:
            result = _read_contiguous_rows(tiff, spec, selection)
            if result is None:
                # Use the same singleton-axis policy as metadata inspection.
                store = tiff.aszarr(series=spec.series, level=spec.level, squeeze=True)
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
        """Read a concrete slice, satisfying Dask's empty metadata probes without I/O."""
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
    """Construct a storage-aligned lazy TIFF array for an explicit series and level.

    The returned Dask graph stores paths and immutable scalar metadata only.
    Computing a slice executes only the source tasks that intersect it; every
    task owns and closes its TIFF and Zarr resources.

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
        A Dask array in the selected level's declared axis order. Strips (and
        tiles as wide as the image) are grouped into row bands of at most
        ``STRIP_BAND_BYTES`` decoded bytes; a larger strip stays one chunk.
        Other tiled levels keep their native tiles.

    Raises
    ------
    IndexError
        If ``series`` or ``level`` is outside the inspected range.
    RasterFormatError
        If a task result disagrees with the inspected shape or dtype.

    Notes
    -----
    Rechunking the result is lazy and does not change the source layer. Each
    source task opens the TIFF and reads its selection either directly (one
    uncompressed contiguous page) or through a short-lived ``tifffile`` Zarr
    store, and closes every resource before returning. A computed selection
    decodes every tile or row band it intersects.
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
    chunks = _task_chunks(selected_level)
    name = f"spatialdata-io-tiff-{tokenize(source, chunks)}"
    data = da.from_array(
        source,
        chunks=chunks,
        name=name,
        lock=False,
        asarray=False,
        fancy=False,
    )
    return LazyRaster(data=data, axes=selected_level.axes)
