"""Build the one-task Dask graph shared by the PNG and JPEG readers.

Neither codec offers bounded spatial selection here, so the whole frame is one
chunk whose task memory-maps the file, decodes it with ``imagecodecs`` and
checks the result against the inspected shape and dtype.
"""

from __future__ import annotations

import mmap
from typing import TYPE_CHECKING

import imagecodecs
import numpy as np
from dask.array.core import Array
from dask.base import tokenize

from spatialdata_io.readers._utils.errors import RasterFormatError
from spatialdata_io.readers._utils.image._types import (
    LazyRaster,
    _EncodedHeader,
    _EncodedReadSpec,
)

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.paths import ArtifactFile


def validate_decoded_size(metadata: _EncodedHeader) -> None:
    """Reject dimensions whose decoded allocation cannot fit in a NumPy array."""
    elements = 1
    for size in metadata.shape:
        if size <= 0:
            message = f"Encoded raster dimensions must be positive; found {metadata.shape}."
            raise RasterFormatError(message)
        elements *= size
    if elements > np.iinfo(np.intp).max // metadata.dtype.itemsize:
        message = f"Decoded raster allocation overflows platform address space: {metadata.shape}, {metadata.dtype}."
        raise RasterFormatError(message)


def _decode_encoded(spec: _EncodedReadSpec) -> NDArray[np.generic]:
    """Decode one PNG or JPEG file and check it against the inspected shape and dtype."""
    try:
        with (
            spec.path.open("rb") as stream,
            mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as encoded,
        ):
            result = imagecodecs.png_decode(encoded) if spec.codec == "png" else imagecodecs.jpeg8_decode(encoded)
    except Exception as error:
        error.add_note(f"Failed to decode {spec.codec.upper()} source {spec.path}.")
        raise

    if result.shape != spec.shape or result.dtype != spec.dtype:
        message = (
            f"Decoded {spec.codec.upper()} result for {spec.path} does not match "
            "inspected metadata: "
            f"expected shape={spec.shape}, dtype={spec.dtype}; "
            f"got shape={result.shape}, dtype={result.dtype}."
        )
        raise RasterFormatError(message)
    return result


def encoded_result(path: ArtifactFile, codec: Literal["png", "jpeg"], metadata: _EncodedHeader) -> LazyRaster:
    """Construct a one-task lazy array for a previously inspected encoded image."""
    spec = _EncodedReadSpec(
        path=path,
        codec=codec,
        shape=metadata.shape,
        dtype=metadata.dtype,
    )
    name = f"spatialdata-io-{codec}-{tokenize(spec)}"
    key = (name, *(0 for _ in metadata.shape))
    graph = {key: (_decode_encoded, spec)}
    chunks = tuple((size,) for size in metadata.shape)
    meta = np.empty((0,) * len(metadata.shape), dtype=metadata.dtype)
    data = Array(graph, name, chunks, dtype=metadata.dtype, meta=meta)
    return LazyRaster(data=data, axes=metadata.axes, color=metadata.color)
