"""Inspect PNG headers and read PNG pixels as one delayed whole-frame task.

A PNG starts with an 8-byte signature and an ``IHDR`` chunk (width, height, bit
depth, color type, interlace), followed by chunks up to the first ``IDAT``::

    signature | IHDR | [PLTE] [tRNS] [ancillary ...] | IDAT ... | IEND

Inspection reads only those headers, skipping ancillary payloads, and derives
the decoded shape, dtype and declared sample colours. One sample per pixel
decodes to ``(y, x)``, several to ``(y, x, s)`` with an interleaved samples axis
``s``. 16-bit images decode to uint16, all others to uint8. Unknown critical
chunks are rejected.

The colour type declares the samples; palette entries are RGB triples, and a
``tRNS`` chunk adds an alpha sample::

    colour type          samples (without / with tRNS)    color
    0 grayscale          gray / gray, alpha               None
    2 RGB                R, G, B / R, G, B, A             "rgb" / "rgba"
    3 palette            R, G, B / R, G, B, A             "rgb" / "rgba"
    4 grayscale + alpha  gray, alpha                      None
    6 RGB + alpha        R, G, B, A                       "rgba"

See the PNG specification: https://www.w3.org/TR/png/
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np

from spatialdata_io.readers._utils.errors import RasterFormatError, add_reader_context
from spatialdata_io.readers._utils.image._encoded import encoded_result, validate_decoded_size
from spatialdata_io.readers._utils.image._types import LazyRaster, _EncodedHeader

if TYPE_CHECKING:
    from pathlib import Path
    from typing import BinaryIO

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.image._types import SampleColor
    from spatialdata_io.readers._utils.paths import ArtifactFile

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IHDR_SIZE = 13
_MAX_PALETTE_SIZE = 768
_GRAYSCALE = 0
_RGB = 2
_INDEXED = 3
_GRAYSCALE_ALPHA = 4
_RGBA = 6
_UINT16_DEPTH = 16
_ASCII_LETTERS = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
_VALID_DEPTHS = {
    _GRAYSCALE: frozenset({1, 2, 4, 8, _UINT16_DEPTH}),
    _RGB: frozenset({8, _UINT16_DEPTH}),
    _INDEXED: frozenset({1, 2, 4, 8}),
    _GRAYSCALE_ALPHA: frozenset({8, _UINT16_DEPTH}),
    _RGBA: frozenset({8, _UINT16_DEPTH}),
}
_BASE_CHANNELS = {_GRAYSCALE: 1, _RGB: 3, _INDEXED: 3, _GRAYSCALE_ALPHA: 2, _RGBA: 4}


def _read_exact(stream: BinaryIO, size: int, *, context: str) -> bytes:
    """Read exactly ``size`` bytes, raising ``RasterFormatError`` on truncation."""
    data = stream.read(size)
    if len(data) != size:
        message = f"Truncated PNG while reading {context}."
        raise RasterFormatError(message)
    return data


def _validate_transparency(color_type: int, payload_size: int, palette_entries: int) -> None:
    """Require the ``tRNS`` payload size to match the color type and palette length."""
    expected = {0: 2, 2: 6}.get(color_type)
    valid = (
        payload_size == expected
        if expected is not None
        else color_type == _INDEXED and 0 < payload_size <= palette_entries
    )
    if not valid:
        message = f"PNG has invalid tRNS length {payload_size} for color type {color_type}."
        raise RasterFormatError(message)


def _png_output_metadata(
    path: Path,
    ihdr: tuple[int, int, int, int],
    *,
    has_palette: bool,
    has_transparency: bool,
) -> _EncodedHeader:
    """Derive the decoded shape, dtype, axes and sample colours from validated ``IHDR`` fields.

    ``ihdr`` holds width, height, bit depth and color type. A ``tRNS`` chunk adds an
    alpha sample to grayscale, RGB and palette images.
    """
    width, height, bit_depth, color_type = ihdr
    valid_depths = _VALID_DEPTHS.get(color_type)
    if valid_depths is None or bit_depth not in valid_depths:
        message = f"PNG {path} has unsupported color type {color_type} and bit depth {bit_depth}."
        raise RasterFormatError(message)
    if color_type == _INDEXED and not has_palette:
        message = f"Indexed PNG {path} has no PLTE chunk before image data."
        raise RasterFormatError(message)

    channels = _BASE_CHANNELS[color_type]
    if has_transparency and color_type in {_GRAYSCALE, _RGB, _INDEXED}:
        channels += 1
    shape = (height, width) if channels == 1 else (height, width, channels)
    axes = ("y", "x") if channels == 1 else ("y", "x", "s")
    color: SampleColor | None = None
    if color_type in {_RGB, _INDEXED, _RGBA}:
        color = "rgba" if color_type == _RGBA or has_transparency else "rgb"
    metadata = _EncodedHeader(
        shape=shape,
        dtype=np.dtype(np.uint16 if bit_depth == _UINT16_DEPTH else np.uint8),
        axes=axes,
        color=color,
    )
    validate_decoded_size(metadata)
    return metadata


def _parse_ihdr(path: Path, payload: bytes) -> tuple[int, int, int, int]:
    """Unpack the ``IHDR`` payload into width, height, bit depth and color type.

    The compression and filter methods must be 0 and the interlace method 0 or 1.
    """
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
    if compression != 0 or filtering != 0 or interlace not in {0, 1}:
        message = f"PNG {path} uses unsupported IHDR methods."
        raise RasterFormatError(message)
    return width, height, bit_depth, color_type


def _parse_optional_chunk(
    path: Path,
    chunk_type: bytes,
    length: int,
    ihdr: tuple[int, int, int, int],
    state: tuple[int, bool],
) -> tuple[int, bool]:
    """Validate one chunk preceding the first ``IDAT``.

    ``state`` and the return value are the palette entry count and whether ``tRNS`` was
    seen. Unknown critical chunks and repeated ``IHDR``, ``PLTE`` or ``tRNS`` chunks are
    rejected.
    """
    palette_entries, has_transparency = state
    if any(byte not in _ASCII_LETTERS for byte in chunk_type):
        message = f"PNG {path} contains an invalid chunk type {chunk_type!r}."
        raise RasterFormatError(message)
    if chunk_type[2] & 0x20:
        message = f"PNG {path} contains a chunk with a lowercase reserved type byte."
        raise RasterFormatError(message)
    if chunk_type == b"IHDR":
        message = f"PNG {path} contains more than one IHDR chunk."
        raise RasterFormatError(message)
    if chunk_type == b"PLTE":
        color_type = ihdr[3]
        maximum_palette_size = min(_MAX_PALETTE_SIZE, 3 * 2 ** ihdr[2])
        if (
            color_type in {_GRAYSCALE, _GRAYSCALE_ALPHA}
            or palette_entries
            or length == 0
            or length % 3
            or length > maximum_palette_size
        ):
            message = f"PNG {path} has an invalid PLTE chunk length {length}."
            raise RasterFormatError(message)
        palette_entries = length // 3
    elif chunk_type == b"tRNS":
        if has_transparency:
            message = f"PNG {path} contains more than one tRNS chunk."
            raise RasterFormatError(message)
        _validate_transparency(ihdr[3], length, palette_entries)
        has_transparency = True
    elif not chunk_type[0] & 0x20:
        message = f"PNG {path} contains unsupported critical chunk {chunk_type!r}."
        raise RasterFormatError(message)
    return palette_entries, has_transparency


def inspect_png_header(path: Path) -> _EncodedHeader:
    """Inspect PNG metadata without decoding pixels or retaining ancillary data."""
    with path.open("rb") as stream:
        if _read_exact(stream, len(_PNG_SIGNATURE), context="signature") != _PNG_SIGNATURE:
            message = f"File does not contain a PNG signature: {path}"
            raise RasterFormatError(message)
        header = _read_exact(stream, 8, context="IHDR chunk header")
        length, chunk_type = struct.unpack(">I4s", header)
        if chunk_type != b"IHDR":
            message = f"PNG {path} does not start with an IHDR chunk."
            raise RasterFormatError(message)
        if length != _IHDR_SIZE:
            message = f"PNG {path} has an invalid IHDR chunk length {length}."
            raise RasterFormatError(message)
        payload = _read_exact(stream, _IHDR_SIZE, context="IHDR payload")
        _read_exact(stream, 4, context="chunk CRC")
        ihdr = _parse_ihdr(path, payload)
        palette_entries = 0
        has_transparency = False
        while True:
            header = _read_exact(stream, 8, context="chunk header")
            length, chunk_type = struct.unpack(">I4s", header)
            if chunk_type == b"IDAT":
                break
            if chunk_type == b"IEND":
                message = f"PNG {path} ended before its first IDAT chunk."
                raise RasterFormatError(message)
            palette_entries, has_transparency = _parse_optional_chunk(
                path,
                chunk_type,
                length,
                ihdr,
                (palette_entries, has_transparency),
            )
            stream.seek(length, 1)
            _read_exact(stream, 4, context="chunk CRC")
    return _png_output_metadata(
        path,
        ihdr,
        has_palette=palette_entries > 0,
        has_transparency=has_transparency,
    )


def read_png(path: ArtifactFile, /, *, context: ReaderErrorContext) -> LazyRaster:
    """Read PNG pixels as one delayed whole-frame task.

    Metadata inspection is eager and skips ancillary payloads; pixel decoding is lazy. Computing
    any selection decodes the complete PNG frame because PNG is not a tiled
    source format at this boundary.

    Parameters
    ----------
    path
        Validated regular PNG file. Its byte signature and header are
        validated here; file names and extensions remain reader concerns.
    context
        Reader and artifact identity added to inspection failures.

    Returns
    -------
    LazyRaster
        One whole-frame Dask source chunk in exact decoder axis order, with
        the sample colours its colour type declares.

    Raises
    ------
    RasterFormatError
        If the header is malformed or describes an unsupported PNG.
    OSError
        If the file cannot be read.

    Notes
    -----
    Inspection failures keep their type and gain a note naming ``context``
    and ``path``.
    """
    try:
        header = inspect_png_header(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=context, path=path)
        raise
    return encoded_result(path, "png", header)
