"""Inspect JPEG frame headers and read JPEG pixels as one delayed whole-frame task.

A JPEG is a sequence of ``0xFF``-prefixed markers: ``SOI``, metadata segments
(``APPn``, ``DQT``, ``DHT``, ...), one start-of-frame (``SOFn``) segment with
precision, height, width and component count, then the scans. Inspection walks
the segment lengths up to the first ``SOF`` without reading entropy-coded data.

Only 8-bit baseline, extended sequential and progressive Huffman frames
(``SOF0``-``SOF2``) with one (``y, x``) or three (``y, x, s``, an interleaved
samples axis) components are supported; they decode to uint8. The decoder
returns three components as R, G, B samples (JFIF stores them as YCbCr and the
decoder converts them), so they declare ``color="rgb"``; one component is
grayscale. EXIF orientation is never applied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from spatialdata_io.readers._utils.errors import RasterFormatError, add_reader_context
from spatialdata_io.readers._utils.image._encoded import encoded_result, validate_decoded_size
from spatialdata_io.readers._utils.image._types import LazyRaster, _EncodedHeader

if TYPE_CHECKING:
    from pathlib import Path
    from typing import BinaryIO

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.paths import ArtifactFile

_SUPPORTED_SOF = frozenset({0xC0, 0xC1, 0xC2})
_ALL_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
_STANDALONE_MARKERS = frozenset({0x01, *range(0xD0, 0xD8)})
_SEGMENT_LENGTH_BYTES = 2
_SOF_FIXED_BYTES = 6
_SUPPORTED_PRECISION = 8


def _read_exact(stream: BinaryIO, size: int, *, context: str) -> bytes:
    """Read exactly ``size`` bytes, raising ``RasterFormatError`` on truncation."""
    data = stream.read(size)
    if len(data) != size:
        message = f"Truncated JPEG while reading {context}."
        raise RasterFormatError(message)
    return data


def _next_marker(stream: BinaryIO) -> int:
    """Return the next marker code, skipping ``0xFF`` fill bytes before it."""
    if _read_exact(stream, 1, context="marker prefix") != b"\xff":
        message = "JPEG contains non-marker data before the first scan."
        raise RasterFormatError(message)
    marker = _read_exact(stream, 1, context="marker")
    while marker == b"\xff":
        marker = _read_exact(stream, 1, context="marker padding")
    if marker == b"\x00":
        message = "JPEG contains a stuffed zero byte before the first scan."
        raise RasterFormatError(message)
    return marker[0]


def _sof_metadata(path: Path, marker: int, payload: bytes) -> _EncodedHeader:
    """Validate a start-of-frame segment and return the decoded shape, dtype, axes and colours."""
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
    if components == 1:
        metadata = _EncodedHeader(shape=(height, width), dtype=np.dtype(np.uint8), axes=("y", "x"), color=None)
    else:
        metadata = _EncodedHeader(
            shape=(height, width, components), dtype=np.dtype(np.uint8), axes=("y", "x", "s"), color="rgb"
        )
    validate_decoded_size(metadata)
    return metadata


def inspect_jpeg_header(path: Path) -> _EncodedHeader:
    """Inspect JPEG frame metadata without decoding pixels or retaining ancillary data."""
    with path.open("rb") as stream:
        if _read_exact(stream, 2, context="SOI") != b"\xff\xd8":
            message = f"File does not contain a JPEG SOI marker: {path}"
            raise RasterFormatError(message)
        while True:
            marker = _next_marker(stream)
            if marker in _STANDALONE_MARKERS:
                continue
            if marker in {0xD8, 0xD9, 0xDA}:
                message = f"JPEG {path} reached marker 0xFF{marker:02X} before a supported SOF marker."
                raise RasterFormatError(message)

            length = int.from_bytes(_read_exact(stream, _SEGMENT_LENGTH_BYTES, context="segment length"), "big")
            if length < _SEGMENT_LENGTH_BYTES:
                message = f"JPEG {path} has invalid segment length {length} at marker 0xFF{marker:02X}."
                raise RasterFormatError(message)
            payload_size = length - _SEGMENT_LENGTH_BYTES
            if marker in _ALL_SOF:
                payload = _read_exact(stream, payload_size, context=f"0xFF{marker:02X} segment")
                return _sof_metadata(path, marker, payload)
            stream.seek(payload_size, 1)


def read_jpeg(path: ArtifactFile, /, *, context: ReaderErrorContext) -> LazyRaster:
    """Read JPEG pixels as one delayed whole-frame task.

    Metadata inspection is eager and skips ancillary payloads; pixel decoding is lazy. Computing
    any selection decodes the complete JPEG frame. EXIF orientation is not
    applied implicitly.

    Parameters
    ----------
    path
        Validated regular JPEG file. Its byte signature and header are
        validated here; file names and extensions remain reader concerns.
    context
        Reader and artifact identity added to inspection failures.

    Returns
    -------
    LazyRaster
        One whole-frame Dask source chunk in exact decoder axis order; three
        components declare RGB samples.

    Raises
    ------
    RasterFormatError
        If the header is malformed or describes an unsupported JPEG.
    OSError
        If the file cannot be read.

    Notes
    -----
    Inspection failures keep their type and gain a note naming ``context``
    and ``path``.
    """
    try:
        header = inspect_jpeg_header(path)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=context, path=path)
        raise
    return encoded_result(path, "jpeg", header)
