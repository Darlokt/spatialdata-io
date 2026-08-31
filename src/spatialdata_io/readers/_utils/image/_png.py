from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np

from spatialdata_io.readers._utils.errors import RasterFormatError
from spatialdata_io.readers._utils.image._encoded import (
    MAX_HEADER_BYTES,
    encoded_result,
    validate_decoded_size,
)
from spatialdata_io.readers._utils.image._path import normalize_file_path
from spatialdata_io.readers._utils.image._types import LazyRaster, _EncodedHeader

if TYPE_CHECKING:
    from pathlib import Path
    from typing import BinaryIO

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
    data = stream.read(size)
    if len(data) != size:
        message = f"Truncated PNG while reading {context}."
        raise RasterFormatError(message)
    return data


def _validate_transparency(color_type: int, payload_size: int, palette_entries: int) -> None:
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
    ihdr: tuple[int, int, int, int] | None,
    *,
    has_palette: bool,
    has_transparency: bool,
) -> _EncodedHeader:
    if ihdr is None:
        message = f"PNG {path} has no IHDR chunk before image data."
        raise RasterFormatError(message)
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
    axes = ("y", "x") if channels == 1 else ("y", "x", "c")
    metadata = _EncodedHeader(
        shape=shape,
        dtype=np.dtype(np.uint16 if bit_depth == _UINT16_DEPTH else np.uint8),
        axes=axes,
    )
    validate_decoded_size(metadata)
    return metadata


def _parse_ihdr(path: Path, payload: bytes) -> tuple[int, int, int, int]:
    if len(payload) != _IHDR_SIZE:
        message = f"PNG {path} has an invalid IHDR chunk length {len(payload)}."
        raise RasterFormatError(message)
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
    if compression != 0 or filtering != 0 or interlace not in {0, 1}:
        message = f"PNG {path} uses unsupported IHDR methods."
        raise RasterFormatError(message)
    return width, height, bit_depth, color_type


def _read_chunk_payload(
    stream: BinaryIO,
    path: Path,
    *,
    chunk_type: bytes,
    length: int,
    scanned: int,
) -> tuple[bytes, int]:
    if scanned + length + 4 > MAX_HEADER_BYTES:
        message = f"PNG metadata before IDAT exceeds the {MAX_HEADER_BYTES}-byte inspection limit: {path}"
        raise RasterFormatError(message)
    context = f"{chunk_type.decode('ascii', errors='replace')} payload"
    payload = _read_exact(stream, length, context=context)
    _read_exact(stream, 4, context="chunk CRC")
    return payload, scanned + length + 4


def _parse_optional_chunk(
    path: Path,
    chunk_type: bytes,
    payload: bytes,
    ihdr: tuple[int, int, int, int],
    state: tuple[int, bool],
) -> tuple[int, bool]:
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
        length = len(payload)
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
        _validate_transparency(ihdr[3], len(payload), palette_entries)
        has_transparency = True
    elif not chunk_type[0] & 0x20:
        message = f"PNG {path} contains unsupported critical chunk {chunk_type!r}."
        raise RasterFormatError(message)
    return palette_entries, has_transparency


def inspect_png_header(path: Path) -> _EncodedHeader:
    """Inspect bounded PNG metadata without decoding pixels."""
    with path.open("rb") as stream:
        if _read_exact(stream, len(_PNG_SIGNATURE), context="signature") != _PNG_SIGNATURE:
            message = f"File does not contain a PNG signature: {path}"
            raise RasterFormatError(message)
        scanned = len(_PNG_SIGNATURE)
        header = _read_exact(stream, 8, context="IHDR chunk header")
        scanned += 8
        length, chunk_type = struct.unpack(">I4s", header)
        if chunk_type != b"IHDR":
            message = f"PNG {path} does not start with an IHDR chunk."
            raise RasterFormatError(message)
        payload, scanned = _read_chunk_payload(
            stream,
            path,
            chunk_type=chunk_type,
            length=length,
            scanned=scanned,
        )
        ihdr = _parse_ihdr(path, payload)
        palette_entries = 0
        has_transparency = False
        while True:
            if scanned + 8 > MAX_HEADER_BYTES:
                message = f"PNG metadata exceeds the {MAX_HEADER_BYTES}-byte inspection limit: {path}"
                raise RasterFormatError(message)
            header = _read_exact(stream, 8, context="chunk header")
            scanned += 8
            length, chunk_type = struct.unpack(">I4s", header)
            if chunk_type == b"IDAT":
                break
            if chunk_type == b"IEND":
                message = f"PNG {path} ended before its first IDAT chunk."
                raise RasterFormatError(message)
            payload, scanned = _read_chunk_payload(
                stream,
                path,
                chunk_type=chunk_type,
                length=length,
                scanned=scanned,
            )
            palette_entries, has_transparency = _parse_optional_chunk(
                path,
                chunk_type,
                payload,
                ihdr,
                (palette_entries, has_transparency),
            )
    return _png_output_metadata(
        path,
        ihdr,
        has_palette=palette_entries > 0,
        has_transparency=has_transparency,
    )


def read_png(path: str | Path) -> LazyRaster:
    """Read PNG pixels as one delayed whole-frame task.

    Metadata inspection is bounded and eager; pixel decoding is lazy. Computing
    any selection decodes the complete PNG frame because PNG is not a tiled
    source format at this boundary.

    Parameters
    ----------
    path
        PNG source. Its byte signature and bounded header are validated; file
        names and extensions remain layout-reader concerns.

    Returns
    -------
    LazyRaster
        One whole-frame Dask source chunk in exact decoder axis order.

    Raises
    ------
    RasterFormatError
        If the bounded header is malformed or describes an unsupported PNG.
    """
    normalized = normalize_file_path(path)
    return encoded_result(normalized, "png", inspect_png_header(normalized))
