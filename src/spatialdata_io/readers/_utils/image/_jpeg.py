from __future__ import annotations

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

_SUPPORTED_SOF = frozenset({0xC0, 0xC1, 0xC2})
_ALL_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
_STANDALONE_MARKERS = frozenset({0x01, *range(0xD0, 0xD8)})
_SEGMENT_LENGTH_BYTES = 2
_SOF_FIXED_BYTES = 6
_SUPPORTED_PRECISION = 8


def _read_exact(stream: BinaryIO, size: int, *, context: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        message = f"Truncated JPEG while reading {context}."
        raise RasterFormatError(message)
    return data


def _next_marker(stream: BinaryIO, *, remaining: int) -> tuple[int, int]:
    if remaining < 1:
        message = "JPEG marker metadata exceeds the bounded inspection limit."
        raise RasterFormatError(message)
    if _read_exact(stream, 1, context="marker prefix") != b"\xff":
        message = "JPEG contains non-marker data before the first scan."
        raise RasterFormatError(message)
    consumed = 1
    if consumed >= remaining:
        message = "JPEG marker metadata exceeds the bounded inspection limit."
        raise RasterFormatError(message)
    marker = _read_exact(stream, 1, context="marker")
    consumed += 1
    while marker == b"\xff":
        if consumed >= remaining:
            message = "JPEG marker metadata exceeds the bounded inspection limit."
            raise RasterFormatError(message)
        marker = _read_exact(stream, 1, context="marker padding")
        consumed += 1
    if marker == b"\x00":
        message = "JPEG contains a stuffed zero byte before the first scan."
        raise RasterFormatError(message)
    return marker[0], consumed


def _sof_metadata(path: Path, marker: int, payload: bytes) -> _EncodedHeader:
    if marker not in _SUPPORTED_SOF:
        message = f"JPEG {path} uses unsupported SOF marker 0xFF{marker:02X}."
        raise RasterFormatError(message)
    if len(payload) < _SOF_FIXED_BYTES:
        message = f"JPEG {path} has a truncated SOF segment."
        raise RasterFormatError(message)
    precision = payload[0]
    height = int.from_bytes(payload[1:3], "big")
    width = int.from_bytes(payload[3:5], "big")
    components = payload[5]
    if precision != _SUPPORTED_PRECISION:
        message = f"JPEG {path} has unsupported sample precision {precision}; expected {_SUPPORTED_PRECISION}."
        raise RasterFormatError(message)
    if components not in {1, 3}:
        message = f"JPEG {path} has unsupported component count {components}; expected 1 or 3."
        raise RasterFormatError(message)
    if len(payload) != _SOF_FIXED_BYTES + 3 * components:
        message = f"JPEG {path} has inconsistent SOF component metadata."
        raise RasterFormatError(message)
    shape = (height, width) if components == 1 else (height, width, components)
    axes = ("y", "x") if components == 1 else ("y", "x", "c")
    metadata = _EncodedHeader(shape=shape, dtype=np.dtype(np.uint8), axes=axes)
    validate_decoded_size(metadata)
    return metadata


def inspect_jpeg_header(path: Path) -> _EncodedHeader:
    """Inspect bounded JPEG frame metadata without decoding pixels."""
    with path.open("rb") as stream:
        if _read_exact(stream, 2, context="SOI") != b"\xff\xd8":
            message = f"File does not contain a JPEG SOI marker: {path}"
            raise RasterFormatError(message)
        scanned = 2
        while scanned <= MAX_HEADER_BYTES:
            marker, marker_bytes = _next_marker(
                stream,
                remaining=MAX_HEADER_BYTES - scanned,
            )
            scanned += marker_bytes
            if marker in _STANDALONE_MARKERS:
                continue
            if marker in {0xD8, 0xD9, 0xDA}:
                message = f"JPEG {path} reached marker 0xFF{marker:02X} before a supported SOF marker."
                raise RasterFormatError(message)

            if scanned + _SEGMENT_LENGTH_BYTES > MAX_HEADER_BYTES:
                message = f"JPEG metadata exceeds the {MAX_HEADER_BYTES}-byte inspection limit: {path}"
                raise RasterFormatError(message)
            length = int.from_bytes(_read_exact(stream, _SEGMENT_LENGTH_BYTES, context="segment length"), "big")
            scanned += _SEGMENT_LENGTH_BYTES
            if length < _SEGMENT_LENGTH_BYTES:
                message = f"JPEG {path} has invalid segment length {length} at marker 0xFF{marker:02X}."
                raise RasterFormatError(message)
            payload_size = length - _SEGMENT_LENGTH_BYTES
            if scanned + payload_size > MAX_HEADER_BYTES:
                message = f"JPEG metadata exceeds the {MAX_HEADER_BYTES}-byte inspection limit: {path}"
                raise RasterFormatError(message)
            context = f"0xFF{marker:02X} segment"
            payload = _read_exact(stream, payload_size, context=context)
            scanned += payload_size
            if marker in _ALL_SOF:
                return _sof_metadata(path, marker, payload)
        message = f"JPEG metadata exceeds the {MAX_HEADER_BYTES}-byte inspection limit: {path}"
        raise RasterFormatError(message)


def read_jpeg(path: str | Path) -> LazyRaster:
    """Read JPEG pixels as one delayed whole-frame task.

    Metadata inspection is bounded and eager; pixel decoding is lazy. Computing
    any selection decodes the complete JPEG frame. EXIF orientation is not
    applied implicitly.

    Parameters
    ----------
    path
        JPEG source. Its SOI marker and bounded frame metadata are validated;
        file names and extensions remain layout-reader concerns.

    Returns
    -------
    LazyRaster
        One whole-frame Dask source chunk in exact decoder axis order.

    Raises
    ------
    RasterFormatError
        If the bounded header is malformed or describes an unsupported JPEG.
    """
    normalized = normalize_file_path(path)
    return encoded_result(normalized, "jpeg", inspect_jpeg_header(normalized))
