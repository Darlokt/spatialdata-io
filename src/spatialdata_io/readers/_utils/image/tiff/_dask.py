"""Build lazy TIFF arrays with source tasks aligned to tiles or whole rows.

Each task holds a path and either a series/level selection or a physical page
index. It opens the TIFF when computed, reads the requested block and closes
the file and any Zarr store before returning a NumPy array.

Whole-width strips are grouped into bands targeting 16 MiB of decoded data.
Where inspection verified contiguous uncompressed rows, one file read can
span many short strips or cover part of a single large strip. Elsewhere the
decoder works in complete storage chunks, so a compressed strip can exceed
the band target. Narrow tiles retain their native geometry. Tasks confined to
one local YX or interleaved YXS tile can call tifffile's segment decoder
directly. Caller-requested output chunks can be applied to the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import tifffile
import zarr
from dask.base import tokenize

from spatialdata_io.readers._utils.errors import RasterFormatError, add_reader_context
from spatialdata_io.readers._utils.image._types import LazyRaster
from spatialdata_io.readers._utils.image.tiff._rows import physical_page, read_contiguous_rows
from spatialdata_io.readers._utils.image.tiff._tiles import read_single_tile

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from tifffile.zarr import ZarrTiffStore

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.image.tiff._types import TiffLevelMetadata, TiffMetadata, _TiffRasterMetadata
    from spatialdata_io.readers._utils.paths import ArtifactFile

type ChunkSelection = tuple[slice, ...]

# Upper bound for the decoded bytes of one task that groups whole-row chunks.
# Every task opens its TIFF, so one task per small strip (often a single row)
# dominates read time. Measured on one-row-strip exports, 16 MiB bands match
# whole-plane conversion speed while keeping partial reads bounded.
STRIP_BAND_BYTES = 16 * 2**20


def _task_chunks(level: _TiffRasterMetadata) -> tuple[int, ...]:
    """Return the chunk geometry of one source task for a TIFF level.

    Levels whose native chunks span every axis after ``Y`` store whole rows:
    strips, or tiles as wide as the image. Consecutive chunks of such levels
    are grouped into row bands of at most ``STRIP_BAND_BYTES`` decoded bytes
    that start at chunk boundaries; a chunk larger than a band stays alone.
    Contiguous uncompressed rows can split a strip into smaller bands. Other
    tiled levels keep their native chunks.
    """
    chunks = level.native_chunks
    row_axis = level.direct_row_axis
    if row_axis is None:
        if "Y" not in level.axes:
            return chunks
        row_axis = level.axes.index("Y")
    trailing = range(row_axis + 1, len(chunks))
    if row_axis == len(chunks) - 1 or any(chunks[axis] != level.shape[axis] for axis in trailing):
        return chunks

    native_rows = chunks[row_axis]
    rows_per_read = 1 if level.direct_row_axis is not None else native_rows
    row_bytes = prod(chunks) // native_rows * level.dtype.itemsize
    band_rows = rows_per_read * max(1, STRIP_BAND_BYTES // (row_bytes * rows_per_read))
    rows = min(level.shape[row_axis], band_rows)
    return (*chunks[:row_axis], rows, *chunks[row_axis + 1 :])


def _check_ome_selection(
    level: tifffile.TiffPageSeries, shape: tuple[int, ...], location: str, selection: ChunkSelection
) -> None:
    # A missing linked page is a fill chunk to Zarr. Check only the
    # page range covered by this task, in the series' storage order.
    first = np.ravel_multi_index(tuple(part.start for part in selection), shape)
    last = np.ravel_multi_index(tuple(part.stop - 1 for part in selection), shape)
    page_size = level.keyframe.size
    for index in range(int(first) // page_size, int(last) // page_size + 1):
        if index >= len(level.pages) or level.pages[index] is None:
            message = f"OME TIFF {location} has a missing image plane {index}."
            raise RasterFormatError(message)


def _read_store(store: ZarrTiffStore, selection: ChunkSelection) -> NDArray[np.generic]:
    try:
        return np.asarray(zarr.open_array(store=store, mode="r")[selection])
    finally:
        store.close()


def _read_tiff_selection(source: _TiffArray, selection: ChunkSelection) -> NDArray[np.generic]:
    """Decode one bounded selection and close every task-owned resource."""
    index = source.index
    location = f"page={index}" if isinstance(index, int) else f"series={index[0]}, level={index[1]}"
    try:
        with tifffile.TiffFile(source.path) as tiff:
            if isinstance(index, int):
                page = physical_page(tiff.pages[index])
                result = (
                    read_contiguous_rows(tiff, page, source.shape, selection)
                    if source.direct_rows
                    else read_single_tile(tiff, page, source.shape, selection)
                )
                if result is None:
                    result = _read_store(page.aszarr(), selection)
            else:
                series, level_index = index
                selected_series = tiff.series[series]
                level = selected_series.levels[level_index]
                if tiff.is_ome:
                    _check_ome_selection(level, source.shape, location, selection)
                result = read_contiguous_rows(tiff, level, source.shape, selection) if source.direct_rows else None
                if result is None and selected_series.transform is None:
                    result = read_single_tile(tiff, level, source.shape, selection)
                if result is None:
                    # Use the same singleton-axis policy as metadata inspection.
                    result = _read_store(tiff.aszarr(series=series, level=level_index, squeeze=True), selection)
    except Exception as error:
        error.add_note(f"Failed TIFF selection path={source.path}, {location}, selection={selection}.")
        raise

    expected_shape = tuple(part.stop - part.start for part in selection)
    if result.shape != expected_shape or result.dtype != source.dtype:
        message = (
            f"TIFF selection result does not match inspected metadata for {source.path}: "
            f"{location}, selection={selection}, "
            f"expected shape={expected_shape}, dtype={source.dtype}; "
            f"got shape={result.shape}, dtype={result.dtype}."
        )
        raise RasterFormatError(message)
    return result


@dataclass(frozen=True, slots=True)
class _TiffArray:
    """Path-backed array adapter used by Dask's compact blockwise layer.

    It holds only the canonical path and immutable selection state, so the
    graph never embeds an open file or store.
    """

    path: ArtifactFile
    # A physical page index or a (series, level) pair.
    index: int | tuple[int, int]
    shape: tuple[int, ...]
    dtype: np.dtype[np.generic]
    direct_rows: bool = False

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
        return _read_tiff_selection(self, selection)


def _select_level(metadata: TiffMetadata, *, series: int, level: int) -> TiffLevelMetadata:
    """Return the inspected storage level of an explicit series and level."""
    if series < 0 or series >= len(metadata.series):
        message = f"TIFF series index {series} is outside [0, {len(metadata.series)})."
        raise IndexError(message)
    selected_series = metadata.series[series]
    if level < 0 or level >= len(selected_series.levels):
        message = f"TIFF level index {level} is outside [0, {len(selected_series.levels)}) for series {series}."
        raise IndexError(message)
    return selected_series.levels[level]


def _select_raster(
    metadata: TiffMetadata, *, series: int | None, level: int | None, page: int | None
) -> tuple[int | tuple[int, int], _TiffRasterMetadata]:
    if page is None:
        if series is None or level is None:
            message = "Select a TIFF page or provide both series and level."
            raise ValueError(message)
        return (series, level), _select_level(metadata, series=series, level=level)
    if series is not None or level is not None:
        message = "A TIFF page selection cannot be combined with series or level."
        raise ValueError(message)
    if metadata.pages is None:
        message = "A TIFF page selection needs metadata inspected with include_pages=True."
        raise ValueError(message)
    if page < 0 or page >= len(metadata.pages):
        message = f"TIFF page index {page} is outside [0, {len(metadata.pages)})."
        raise IndexError(message)
    return page, metadata.pages[page]


def read_tiff(
    metadata: TiffMetadata,
    *,
    series: int | None = None,
    level: int | None = None,
    page: int | None = None,
    context: ReaderErrorContext,
) -> LazyRaster:
    """Construct a storage-aligned lazy TIFF array for a series level or physical page.

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
    page
        Zero-based top-level IFD index in file order. Requires metadata
        inspected with ``include_pages=True``. Select this alone or pass
        both ``series`` and ``level``.
    context
        Reader and artifact identity added to out-of-range selection errors.

    Returns
    -------
    LazyRaster
        A Dask array in the selected level's declared axis order. Strips (and
        tiles as wide as the image) are grouped into row bands of at most
        ``STRIP_BAND_BYTES`` decoded bytes. Contiguous uncompressed storage
        can be read as individual rows or row bands. An encoded strip larger
        than a band or a single row larger than a band stays one chunk.
        Other tiled levels keep their native tiles. ``color`` is the level's
        declared samples colour model.

    Raises
    ------
    IndexError
        If ``series``, ``level`` or ``page`` is outside the inspected range.
    ValueError
        If the selection is ambiguous, missing, or calls for uninspected pages.
    RasterFormatError
        If a task result disagrees with the inspected shape or dtype.

    Notes
    -----
    Rechunking the result is lazy and does not change the source layer. Each
    source task opens the TIFF, reads its selection (contiguous rows, a single
    tile, or through a short-lived ``tifffile`` Zarr store) and closes every
    resource before returning. A computed selection
    decodes every tile or row band it intersects. An out-of-range selection
    keeps its ``IndexError`` type and gains a note naming ``context`` and the
    TIFF path.
    """
    try:
        index, selected = _select_raster(metadata, series=series, level=level, page=page)
    except (IndexError, ValueError) as error:
        add_reader_context(error, context=context, path=metadata.path)
        raise
    source = _TiffArray(metadata.path, index, selected.shape, selected.dtype, selected.direct_row_axis is not None)
    chunks = _task_chunks(selected)
    name = f"spatialdata-io-tiff-{tokenize(source, chunks)}"
    data = da.from_array(
        source,
        chunks=chunks,
        name=name,
        lock=False,
        asarray=False,
        fancy=False,
    )
    return LazyRaster(data=data, axes=selected.axes, color=selected.color)
