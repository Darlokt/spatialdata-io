"""Read post-Xenium H&E or IF images aligned in Xenium Explorer.

The optional alignment CSV is a 3 x 3 row-major similarity matrix ``M`` mapping
image pixel coordinates (x right, y down) to Xenium morphology pixels. In the
micrometre ``global`` frame the image therefore carries
``Affine(diag(ps, ps, 1) @ M)``, with ``ps`` the Xenium pixel size from
``experiment.xenium``, not the image's own pixel size. Without a CSV the image is
taken to be registered to the Xenium pixel grid already and carries ``Scale(ps)``.
https://kb.10xgenomics.com/hc/en-us/articles/35386990499853-How-can-I-convert-coordinates-between-H-E-image-and-Xenium-data

Level 0 of the first TIFF series is read with its declared axes, or the
caller's ``data_axes``::

    YXS    interleaved samples, e.g. an RGB H&E scan
    CYX    channels, e.g. an IF stack or an H&E stored as Hematoxylin/Eosin/Residual
    YX     one channel

Samples become channels named ``r, g, b(, a)`` only when the TIFF declares an
RGB colour model; otherwise channels take their OME names or IDs, and
positional indexes without OME metadata or when ``data_axes`` reads another
axis as the channels. Three channels are never guessed to be RGB. Axes the
file declares that cannot become ``c, y, x`` are malformed content and raise
``ReaderFormatError``; ``data_axes`` that do not fit the file raise
``ValueError`` with a note naming the image.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
from spatialdata.models import Image2DModel
from spatialdata.transformations import Affine, Scale

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.csv import csv_errors
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import (
    RasterOptions,
    inspect_tiff,
    make_raster_options,
    normalize_cyx_image,
    read_tiff,
    validate_data_axes,
)
from spatialdata_io.readers._utils.paths import ArtifactFile, require_file
from spatialdata_io.readers.xenium._schemas import ALIGNMENT_TYPES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray, DataTree

_AXES = ("c", "y", "x")
_MATRIX_SHAPE = (3, 3)
_CONTEXT = ReaderErrorContext("xenium", "aligned image")
_CSV_BLOCK_BYTES = 64 * 1024


def read_alignment(path: ArtifactFile, *, context: ReaderErrorContext) -> NDArray[np.float64]:
    """Read a 3 x 3 alignment matrix without header, with last row ``[0, 0, 1]``.

    Records must fit a 64 KiB parser block. Excess rows fail after bounded
    read-ahead, without materializing the rest of the file.

    Raises
    ------
    ReaderFormatError
        If the file is not three records of three finite numbers, has another
        last row, or is not invertible.
    """
    batches: list[pa.RecordBatch] = []
    rows = 0
    with (
        path.open("rb") as handle,
        csv_errors(context=context, path=path),
        pacsv.open_csv(
            handle,
            read_options=pacsv.ReadOptions(
                column_names=ALIGNMENT_TYPES.names, block_size=_CSV_BLOCK_BYTES, use_threads=False
            ),
            convert_options=pacsv.ConvertOptions(
                column_types=ALIGNMENT_TYPES,
                include_columns=ALIGNMENT_TYPES.names,
                null_values=[],
                strings_can_be_null=False,
            ),
        ) as reader,
    ):
        for batch in reader:
            rows += batch.num_rows
            if rows > _MATRIX_SHAPE[0]:
                raise ReaderFormatError(
                    context=context, path=path, found="more than three records", expected="a 3 x 3 alignment matrix"
                )
            batches.append(batch)
        table = pa.Table.from_batches(batches, schema=reader.schema)
    matrix = np.column_stack([column.to_numpy() for column in table.columns])
    if (
        matrix.shape != _MATRIX_SHAPE
        or not np.isfinite(matrix).all()
        or not np.array_equal(matrix[2], [0.0, 0.0, 1.0])
        or np.linalg.det(matrix[:2, :2]) == 0
    ):
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"matrix {matrix.tolist()}",
            expected="a 3 x 3 comma-separated finite invertible matrix with last row 0,0,1",
        )
    return matrix


def read_aligned(  # noqa: PLR0913, RUF100 - Explicit source interpretation, raster options and context.
    image: ArtifactFile,
    alignment: ArtifactFile | None,
    *,
    pixel_size: float,
    data_axes: str | None,
    raster: RasterOptions,
    context: ReaderErrorContext,
) -> DataArray | DataTree:
    """Return one aligned image as lazy CYX pixels in the micrometre frame.

    Raises
    ------
    ValueError
        If lowercase ``data_axes`` cannot be mapped to CYX for the image (with
        a note naming it), or ``raster`` requests more downsampling than the
        image extent allows.
    ReaderFormatError
        If the image metadata, its declared axes or the alignment matrix are
        malformed.
    """
    metadata = inspect_tiff(image, context=context)
    lazy = read_tiff(metadata, series=0, level=0, context=context)
    data, channels = normalize_cyx_image(lazy, data_axes, tiff=(metadata, 0), context=context, path=image)
    raster.validate_image_pyramid(data, element="the aligned image")

    transformation: BaseTransformation = Scale([pixel_size, pixel_size], axes=("y", "x"))
    if alignment is not None:
        matrix = np.diag([pixel_size, pixel_size, 1.0]) @ read_alignment(
            alignment, context=replace(context, artifact="image alignment")
        )
        transformation = Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))
    return Image2DModel.parse(
        data,
        dims=_AXES,
        c_coords=channels,
        rgb=False,
        scale_factors=raster.model_scale_factors,
        chunks=raster.image_chunks(),
        transformations={"global": transformation},
    )


def xenium_aligned_image(  # noqa: PLR0913, RUF100 - Explicit source interpretation and raster options.
    image_file: str | Path,
    alignment_file: str | Path | None = None,
    *,
    pixel_size: float,
    data_axes: str | None = None,
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
        Axes of the stored array as one string, one distinct letter per
        dimension in any letter case including ``X`` and ``Y``, overriding the
        TIFF axes, for example ``"YXS"``. ``S`` marks interleaved samples and
        ``C`` channels; both become the channel axis, so at most one may
        appear.
    scale_factors
        Integer factors of at least two, one per generated pyramid level; empty
        for a single scale.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes; by default
        one channel per chunk and square 1024-pixel tiles.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        A ``(c, y, x)`` image in the source dtype. Its ``global``
        transformation maps image pixels to micrometres:
        ``Affine(diag(pixel_size, pixel_size, 1) @ M)`` with an alignment
        matrix ``M``, otherwise ``Scale(pixel_size)``. Pixels stay lazy.

        Channel names follow the file's declarations:

        - ``r, g, b`` (and ``a`` for declared alpha) when the channels are the
          TIFF samples and the TIFF declares an RGB colour model
          (photometric RGB or JPEG-compressed YCbCr);
        - otherwise the OME channel names when all are nonempty and unique, else
          the OME channel IDs, when the channel axis is the declared OME channel
          or samples axis; samples without a colour model are named
          ``<name or ID>:sample:<index>``;
        - positional ``0, 1, ...`` without OME metadata, or when ``data_axes``
          reads another axis as the channels.

        Three or four channels are never assumed to be RGB from their count.

    Raises
    ------
    TypeError, ValueError
        If options are invalid, or ``data_axes`` cannot be mapped to CYX for
        the image (a ``ValueError`` with a note naming it).
    FileNotFoundError, IsADirectoryError
        If a file is missing or a directory.
    ValueError
        A contextual ``ReaderFormatError`` if the image, its declared axes, its
        OME channels or the alignment matrix are malformed.
    """
    if isinstance(pixel_size, bool) or not isinstance(pixel_size, int | float):
        message = "pixel_size must be a number of micrometres."
        raise TypeError(message)
    if not math.isfinite(pixel_size) or pixel_size <= 0:
        message = "pixel_size must be a finite positive number of micrometres."
        raise ValueError(message)
    axes = validate_data_axes(data_axes)
    raster = make_raster_options(scale_factors, chunks)

    image = require_file(Path(image_file).expanduser(), context=_CONTEXT)
    alignment = None if alignment_file is None else require_file(Path(alignment_file).expanduser(), context=_CONTEXT)
    return read_aligned(image, alignment, pixel_size=float(pixel_size), data_axes=axes, raster=raster, context=_CONTEXT)
