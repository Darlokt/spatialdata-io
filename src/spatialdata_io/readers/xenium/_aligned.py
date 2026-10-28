"""Read post-Xenium H&E or IF images aligned in Xenium Explorer.

The optional alignment CSV is a 3 x 3 row-major similarity matrix ``M`` mapping
image pixel coordinates (x right, y down) to Xenium morphology pixels. In the
micrometre ``global`` frame the image therefore carries
``Affine(diag(ps, ps, 1) @ M)``, with ``ps`` the Xenium pixel size from
``experiment.xenium``, not the image's own pixel size. Without a CSV the image is
taken to be registered to the Xenium pixel grid already and carries ``Scale(ps)``.
https://kb.10xgenomics.com/hc/en-us/articles/35386990499853-How-can-I-convert-coordinates-between-H-E-image-and-Xenium-data
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from spatialdata.models import Image2DModel
from spatialdata.transformations import Affine, Scale

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image import (
    DEFAULT_CHUNKS,
    DEFAULT_SCALE_FACTORS,
    RasterOptions,
    inspect_tiff,
    make_raster_options,
    normalize_raster_axes,
    read_tiff,
)
from spatialdata_io.readers._utils.paths import ArtifactFile, require_file

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray, DataTree

MAX_ALIGNMENT_BYTES = 64 * 1024
_MATRIX_SHAPE = (3, 3)
_RGB_CHANNELS = ("r", "g", "b", "a")
_RGB_SIZES = (3, 4)


def read_alignment(path: ArtifactFile) -> NDArray[np.float64]:
    """Read a strict 3 x 3 alignment matrix with last row ``[0, 0, 1]``.

    Raises
    ------
    ReaderFormatError
        If the file exceeds 64 KiB, is not three rows of three finite numbers,
        has another last row, or is not invertible.
    """
    context = ReaderErrorContext("xenium", "image alignment")
    expected = "a 3 x 3 comma-separated finite invertible matrix with last row 0,0,1"
    with path.open("rb") as handle:
        payload = handle.read(MAX_ALIGNMENT_BYTES + 1)
    if len(payload) > MAX_ALIGNMENT_BYTES:
        raise ReaderFormatError(context=context, path=path, found="an oversized file", expected=expected)
    try:
        rows = [row for row in csv.reader(payload.decode("utf-8-sig").splitlines(), strict=True) if row]
        matrix = np.asarray([[float(value) for value in row] for row in rows], dtype=np.float64)
    except (UnicodeDecodeError, csv.Error, ValueError) as error:
        raise ReaderFormatError(context=context, path=path, found="non-numeric content", expected=expected) from error
    if (
        matrix.shape != _MATRIX_SHAPE
        or not np.isfinite(matrix).all()
        or not np.array_equal(matrix[2], [0.0, 0.0, 1.0])
        or np.linalg.det(matrix[:2, :2]) == 0
    ):
        raise ReaderFormatError(context=context, path=path, found=f"matrix {matrix.tolist()}", expected=expected)
    return matrix


def read_aligned(
    image: ArtifactFile,
    alignment: ArtifactFile | None,
    *,
    pixel_size: float,
    data_axes: tuple[str, ...] | None,
    raster: RasterOptions,
) -> DataArray | DataTree:
    """Return one aligned image as lazy CYX pixels in the micrometre frame."""
    context = ReaderErrorContext("xenium", "aligned image")
    try:
        metadata = inspect_tiff(image)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=context, path=image)
        raise

    lazy = read_tiff(metadata, series=0, level=0)
    axes = tuple(axis.lower() for axis in (lazy.axes if data_axes is None else data_axes))
    if len(axes) != lazy.data.ndim:
        raise ReaderFormatError(
            context=context,
            path=image,
            found=f"axes {axes} for shape {lazy.data.shape}",
            expected="one axis per dimension",
        )
    # TIFF samples (S) are interleaved RGB(A) components; C axes are named channels.
    samples = "s" in axes
    data = normalize_raster_axes(
        lazy.data, tuple("c" if axis == "s" else axis for axis in axes), target_axes=("c", "y", "x"), add_missing=("c",)
    )

    channels = data.shape[0]
    names: list[str] | None = None
    if samples and channels in _RGB_SIZES:
        names = list(_RGB_CHANNELS[:channels])
    elif metadata.ome is not None and metadata.ome.images:
        ome_names = [channel.name for channel in metadata.ome.images[0].pixels.channels]
        if len(ome_names) == channels and all(ome_names) and len(set(ome_names)) == channels:
            names = [str(name) for name in ome_names]

    transformation: BaseTransformation = Scale([pixel_size, pixel_size], axes=("y", "x"))
    if alignment is not None:
        matrix = np.diag([pixel_size, pixel_size, 1.0]) @ read_alignment(alignment)
        transformation = Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))
    return Image2DModel.parse(
        data,
        dims=("c", "y", "x"),
        c_coords=names,
        rgb=False,
        scale_factors=raster.model_scale_factors,
        chunks=raster.image_chunks(),
        transformations={"global": transformation},
    )


def xenium_aligned_image(
    image_file: str | Path,
    alignment_file: str | Path | None = None,
    *,
    pixel_size: float,
    data_axes: Sequence[str] | None = None,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
) -> DataArray | DataTree:
    """Read an H&E or IF image aligned to a Xenium dataset in Xenium Explorer.

    Parameters
    ----------
    image_file
        Pyramidal OME-TIFF; level 0 of its first series is read lazily.
    alignment_file
        Optional Xenium Explorer ``*_imagealignment.csv``: a 3 x 3 row-major
        similarity matrix from image pixels to Xenium morphology pixels. Without
        it the image must already be registered to the Xenium pixel grid.
    pixel_size
        Xenium morphology pixel size in micrometres, from ``experiment.xenium``
        ``pixel_size`` (``sdata.attrs["xenium"]["experiment"]["pixel_size"]``).
        The image's own OME pixel size is a different quantity and is not used.
    data_axes
        Explicit single-letter axes of the stored array, overriding the TIFF
        axes, for example ``("y", "x", "s")``. ``s`` marks interleaved RGB(A)
        samples and ``c`` named channels.
    scale_factors
        Integer factors of at least two, one per generated pyramid level; empty
        for a single scale.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes; by default
        one channel per chunk and square 1024-pixel tiles.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        A ``(c, y, x)`` image in the source dtype. RGB(A) samples are named
        ``r, g, b(, a)``; other channels use their OME names, or default indexes
        without OME names. Its ``global`` transformation maps image pixels to
        micrometres: ``Affine(diag(pixel_size, pixel_size, 1) @ M)`` with an
        alignment matrix ``M``, otherwise ``Scale(pixel_size)``. Pixels stay lazy.

    Raises
    ------
    TypeError, ValueError
        If options are invalid.
    FileNotFoundError, IsADirectoryError
        If a file is missing or a directory.
    ValueError
        A contextual ``ReaderFormatError`` if the image or the alignment matrix
        is malformed.
    """
    if isinstance(pixel_size, bool) or not isinstance(pixel_size, int | float):
        message = "pixel_size must be a number of micrometres."
        raise TypeError(message)
    if not math.isfinite(pixel_size) or pixel_size <= 0:
        message = "pixel_size must be a finite positive number of micrometres."
        raise ValueError(message)
    if data_axes is not None and (isinstance(data_axes, str) or not isinstance(data_axes, Sequence)):
        message = "data_axes must be a sequence of single-letter axis names."
        raise TypeError(message)
    raster = make_raster_options(scale_factors, chunks)

    context = ReaderErrorContext("xenium", "aligned image")
    image = require_file(Path(image_file).expanduser(), context=context)
    alignment = None if alignment_file is None else require_file(Path(alignment_file).expanduser(), context=context)
    axes = None if data_axes is None else tuple(data_axes)
    return read_aligned(image, alignment, pixel_size=float(pixel_size), data_axes=axes, raster=raster)
