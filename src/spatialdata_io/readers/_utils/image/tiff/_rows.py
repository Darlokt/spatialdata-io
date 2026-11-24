"""Resolve direct byte reads for bands of uncompressed TIFF rows.

A YX plane, or a YXS plane with interleaved samples, can expose whole rows
as consecutive bytes. The storage checks establish a local, in-file extent
and require the bytes to be in final pixel order. For a series, the selected
plane may have its own physical directory or share an ImageJ pixel run with
other planes. The latter requires a verified extent for the whole stack.

The Dask task owns the open TIFF. For an eligible selection, one seek and
read returns the selected rows in native byte order. Ineligible selections
fall back to the TIFF decoder.
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import numpy as np
import tifffile

if TYPE_CHECKING:
    from numpy.typing import NDArray

type TiffImage = tifffile.TiffPage | tifffile.TiffPageSeries


def physical_page(page: tifffile.TiffPage | tifffile.TiffFrame) -> tifffile.TiffPage:
    """Read a full IFD when tifffile cached only a frame with inherited tags."""
    return page.aspage() if isinstance(page, tifffile.TiffFrame) else page


def _page_row_offset(
    tiff: tifffile.TiffFile, stored: tifffile.TiffPage | tifffile.TiffFrame | None, expected: tifffile.TiffPage
) -> int | None:
    if stored is None or stored.parent is not tiff or (isinstance(stored, tifffile.TiffFrame) and stored.is_virtual):
        return None
    page = physical_page(stored)
    if (
        page.axes in {"YX", "YXS"}
        and page.axes == expected.axes
        and page.shape == expected.shape
        and page.dtype is not None
        and page.dtype == expected.dtype
        and page.is_final
        and page.dataoffsets[0] > 0
        and sum(page.databytecounts) == page.nbytes
        and page.dataoffsets[0] + page.nbytes <= tiff.filehandle.size
    ):
        return page.dataoffsets[0]
    return None


def _imagej_stack_offset(tiff: tifffile.TiffFile, level: tifffile.TiffPageSeries) -> int | None:
    """Check an ImageJ run with all its pixels stored before any subsequent IFD."""
    if level.kind != "imagej" or (offset := _page_row_offset(tiff, level[0], level.keyframe)) is None:
        return None
    end = offset + level.nbytes
    if end > tiff.filehandle.size or (tiff.pages.is_multipage and end > tiff.pages[1].offset):
        return None
    return offset


def direct_row_axis(tiff: tifffile.TiffFile, image: TiffImage, shape: tuple[int, ...]) -> int | None:
    """Locate the physical row axis when its tasks can bypass strip decoding."""
    page = image.keyframe
    if page.axes not in {"YX", "YXS"} or not page.is_final or shape[-page.ndim :] != page.shape:
        return None
    row_axis = len(shape) - page.ndim
    if isinstance(image, tifffile.TiffPage):
        eligible = _page_row_offset(tiff, image, page) is not None
    elif image.transform is not None or image.dtype != page.dtype:
        return None
    elif row_axis == 0:
        eligible = len(image) == 1 and _page_row_offset(tiff, image[0], page) is not None
    else:
        # Ordinary multi-page files with several strips per plane retain the
        # native strip grid. Splitting strips requires direct access to every plane.
        if page.is_tiled or len(page.dataoffsets) != 1:
            return None
        eligible = _imagej_stack_offset(tiff, image) is not None or (
            len(image) == prod(shape[:row_axis])
            and all(_page_row_offset(tiff, stored, page) is not None for stored in image)
        )
    return row_axis if eligible else None


def read_contiguous_rows(
    tiff: tifffile.TiffFile,
    image: TiffImage,
    shape: tuple[int, ...],
    selection: tuple[slice, ...],
) -> NDArray[np.generic] | None:
    """Read a band of whole rows from one uncompressed plane in a single file read.

    A TIFF writer can store each row as its own strip or an entire plane as a
    single strip. When the selected rows occupy consecutive bytes in file
    order, a single seek and read avoids per-strip Zarr decoding and can read
    just part of a large strip. ImageJ stacks can store several such planes
    under one physical IFD. Resolve the selected plane before seeking.

    The caller owns the open TIFF. Return an independent native-endian array,
    or ``None`` when the selection requires the ordinary TIFF decoder.
    """
    page = image.keyframe
    if page.dtype is None or shape[-page.ndim :] != page.shape:
        return None
    row_axis = len(shape) - page.ndim
    if any(part.stop - part.start != 1 for part in selection[:row_axis]) or any(
        (part.start, part.stop) != (0, size)
        for part, size in zip(selection[row_axis + 1 :], page.shape[1:], strict=True)
    ):
        return None
    if isinstance(image, tifffile.TiffPage):
        offset = _page_row_offset(tiff, image, page)
    else:
        if image.transform is not None or image.dtype != page.dtype:
            return None
        plane = int(np.ravel_multi_index(tuple(part.start for part in selection[:row_axis]), shape[:row_axis]))
        stack_offset = _imagej_stack_offset(tiff, image) if row_axis else None
        offset = (
            _page_row_offset(tiff, image[plane], page) if stack_offset is None else stack_offset + plane * page.nbytes
        )
    if offset is None:
        return None
    rows = selection[row_axis]
    row_items = prod(page.shape[1:])
    # FileHandle.read_array interprets file order and returns native byte order.
    dtype = np.dtype(tiff.byteorder + page.dtype.char)
    tiff.filehandle.seek(offset + rows.start * row_items * dtype.itemsize)
    data = tiff.filehandle.read_array(dtype, count=(rows.stop - rows.start) * row_items)
    return data.reshape(tuple(part.stop - part.start for part in selection))
