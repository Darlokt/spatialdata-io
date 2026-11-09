"""Normalize a 2D raster to ``c, y, x`` and name its channels by the shared rules.

Several readers return one 2D image whose axes come either from the file or
from a caller's ``data_axes`` override, for example ``"CYX"`` for a stack
without axis metadata. Both readings follow one contract:

- interleaved samples ``S`` become the channel axis, like ``C``;
- singleton axes absent from ``c, y, x`` are dropped, a missing ``C`` is added
  as a singleton, and any other extra axis fails;
- a failure is malformed content when the file declared the axes
  (``ReaderFormatError`` chained from the normalization error) and a caller
  error when ``data_axes`` named them (the ``ValueError`` itself, with a note
  naming the reader context and file);
- channel names follow ``channel_source``: ``r, g, b(, a)`` for declared RGB(A)
  samples, else OME names when the channel axis is the declared OME channel or
  samples axis, else ``None`` for positional names.

Readers keep their file and series selection, contexts, pyramid validation,
transformations and element names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.image._axes import normalize_raster_axes
from spatialdata_io.readers._utils.image._channels import channel_source, raster_channel_names
from spatialdata_io.readers._utils.image._ome import ome_channel_names

if TYPE_CHECKING:
    from pathlib import Path

    import dask.array as da

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.image._types import LazyRaster
    from spatialdata_io.readers._utils.image.tiff import TiffMetadata

_CYX = ("c", "y", "x")
_EXPECTED_AXES = "Y and X axes with at most one channel or samples axis and only singleton other axes"


def normalize_cyx_image(
    raster: LazyRaster,
    data_axes: str | None,
    /,
    *,
    tiff: tuple[TiffMetadata, int] | None,
    context: ReaderErrorContext,
    path: Path,
) -> tuple[da.Array, list[str] | None]:
    """Return a raster as lazy ``c, y, x`` pixels and its channel coordinates.

    Parameters
    ----------
    raster
        Lazy raster from ``read_tiff``, ``read_png`` or ``read_jpeg``.
    data_axes
        A caller's override of the stored axes, validated by
        ``validate_data_axes`` (any letter case), or ``None`` for the axes the
        file declares.
    tiff
        Inspected metadata and selected series of a TIFF source, whose OME
        channel records may name the channels; ``None`` for PNG and JPEG.
    context
        Owning reader and artifact, for both axis and OME channel errors.
    path
        The raster's file, named in errors.

    Returns
    -------
    tuple of dask.array.Array and list of str or None
        The lazy ``c, y, x`` view, built only from singleton removal and
        insertion and transposition, and its channel coordinates: ``r, g, b``
        or ``r, g, b, a`` for declared RGB(A) samples, OME channel names or IDs
        (``<name or ID>:sample:<k>`` for non-RGB OME samples) when the channel
        axis is the declared OME channel or samples axis, else ``None`` for
        positional ``0..n-1``.

    Raises
    ------
    ReaderFormatError
        If the axes the file declares cannot become ``c, y, x`` (chained from
        the normalization error), or the OME channel records that describe
        the channel axis are corrupt (a count mismatch or repeated IDs).
    ValueError
        If ``data_axes`` does not fit the file (another rank, or a
        non-singleton axis outside ``c, y, x``), with a note naming
        ``context`` and ``path``.

    Notes
    -----
    No pixels are read. OME records are consulted only when they describe the
    channel axis, so a ``data_axes`` reinterpretation of another axis (for
    example Z) as channels gets positional names, not names of another axis.
    """
    declared = tuple(axis.lower() for axis in raster.axes)
    source = declared if data_axes is None else tuple(data_axes.lower())
    try:
        data = normalize_raster_axes(
            raster.data, tuple("c" if axis == "s" else axis for axis in source), target_axes=_CYX, add_missing=("c",)
        )
    except ValueError as error:
        if data_axes is not None:
            add_reader_context(error, context=context, path=path)
            raise
        raise ReaderFormatError(context=context, path=path, found=str(error), expected=_EXPECTED_AXES) from error

    samples, ome_samples = channel_source(declared, source)
    ome = None
    if tiff is not None and ome_samples is not None:
        metadata, series = tiff
        ome = ome_channel_names(metadata, series=series, count=int(data.shape[0]), samples=ome_samples, context=context)
    return data, raster_channel_names(raster.color, samples=samples, ome=ome)
