"""Read selections confined to one tile of an ordinary local TIFF plane."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import numpy as np
import tifffile

from spatialdata_io.readers._utils.errors import RasterFormatError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.image.tiff._rows import TiffImage


def read_single_tile(
    tiff: tifffile.TiffFile, image: TiffImage, shape: tuple[int, ...], selection: tuple[slice, ...]
) -> NDArray[np.generic] | None:
    """Decode one local YX or interleaved YXS tile with tifffile's segment decoder.

    Return None for selections whose storage mapping needs the TIFF Zarr
    decoder. Each leading logical dimension addresses physical planes in
    series order. The plane's own file directory supplies the tile offsets
    and JPEG tables. The caller owns the open TIFF.
    """
    keyframe = image.keyframe
    if (
        not keyframe.is_tiled
        or keyframe.axes not in {"YX", "YXS"}
        or shape[-keyframe.ndim :] != keyframe.shape
        or keyframe.jpegheader is not None
    ):
        return None
    plane_ndim = len(shape) - keyframe.ndim
    prefix, pixels = selection[:plane_ndim], selection[plane_ndim:]
    rows, columns = pixels[:2]
    tile_height, tile_width = keyframe.tilelength, keyframe.tilewidth
    tile_y, tile_x = rows.start // tile_height, columns.start // tile_width
    if (rows.stop - 1) // tile_height != tile_y or (columns.stop - 1) // tile_width != tile_x:
        return None
    if isinstance(image, tifffile.TiffPage):
        page = image
    else:
        if (
            image.transform is not None
            or image.dtype != keyframe.dtype
            or image.dataoffset is not None
            or len(image) != prod(shape[:plane_ndim])
            or any(part.stop - part.start != 1 for part in prefix)
        ):
            return None
        plane = int(np.ravel_multi_index(tuple(part.start for part in prefix), shape[:plane_ndim]))
        page = image[plane]

    tile_index = tile_y * ((keyframe.imagewidth + tile_width - 1) // tile_width) + tile_x
    if (
        page is None
        or page.parent is not tiff
        or page.keyframe is not keyframe
        or (isinstance(page, tifffile.TiffFrame) and page.is_virtual)
        or tile_index >= min(len(page.dataoffsets), len(page.databytecounts))
    ):
        return None
    offset, count = page.dataoffsets[tile_index], page.databytecounts[tile_index]
    if offset == 0 or count == 0:
        # Preserve the Zarr fill behavior for empty or missing tiles.
        return None
    handle = tiff.filehandle
    if offset + count > handle.size:
        message = f"TIFF tile {tile_index} of IFD {page.index} extends past the end of the file."
        raise RasterFormatError(message)
    # Read the payload before decoder initialization can seek for codec metadata.
    handle.seek(offset)
    encoded = handle.read(count)
    tile, _, _ = keyframe.decode(encoded, tile_index, jpegtables=page.jpegtables)
    if tile is None:
        message = f"TIFF tile {tile_index} of IFD {page.index} has a payload but decoded no pixels."
        raise RasterFormatError(message)
    y_offset, x_offset = tile_y * tile_height, tile_x * tile_width
    # Decoded tiles have dimensions (depth, rows, columns, samples). Edge
    # tiles may omit padding beyond the image.
    block = tile[
        0,
        slice(rows.start - y_offset, rows.stop - y_offset),
        slice(columns.start - x_offset, columns.stop - x_offset),
        pixels[2] if keyframe.axes == "YXS" else 0,
    ]
    return block.reshape((1,) * plane_ndim + block.shape)
