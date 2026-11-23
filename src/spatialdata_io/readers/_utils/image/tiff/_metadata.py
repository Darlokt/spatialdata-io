"""Inspect TIFF storage and semantic metadata without decoding pixels.

Besides storage geometry, inspection records the colour model each level
declares for its samples axis ``S``, from the TIFF 6.0 baseline tags
``PhotometricInterpretation`` and ``ExtraSamples`` of its first page::

    photometric           ExtraSamples                 decoded S     color
    2 RGB                 none                         R, G, B       "rgb"
    2 RGB                 1 associated alpha or        R, G, B, A    "rgba"
                          2 unassociated alpha
    6 YCbCr, JPEG         none, interleaved samples    R, G, B       "rgb"
    anything else         -                            unchanged     None

"Anything else" includes ``MinIsBlack`` samples, RGB with an unspecified (0)
or a second extra sample, uncompressed or planar YCbCr, and palette images.
The JPEG codec converts interleaved YCbCr to RGB on decoding (TIFF Technical
Note 2), and ``tifffile`` itself stores JPEG-compressed RGB data as YCbCr, so
that combination declares RGB samples too. Other YCbCr data decodes to luma and
chroma samples.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import tifffile
import zarr
from ome_types import from_xml

from spatialdata_io.readers._utils.errors import RasterFormatError, add_reader_context
from spatialdata_io.readers._utils.image.tiff._types import (
    TiffLevelMetadata,
    TiffMetadata,
    TiffSeriesMetadata,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ome_types import OME
    from ome_types.model import Pixels, TiffData

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.image._types import SampleColor
    from spatialdata_io.readers._utils.paths import ArtifactFile

_TIFF_SIGNATURES = frozenset({b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"})
# Old-style, new-style, lossy (DNG) and Aperio JPEG: the compressions whose
# YCbCr data tifffile decodes to RGB.
_JPEG_COMPRESSIONS = frozenset({6, 7, 33007, 34892})
_ALPHA_SAMPLES = frozenset({tifffile.EXTRASAMPLE.ASSOCALPHA, tifffile.EXTRASAMPLE.UNASSALPHA})


def _has_tiff_signature(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(4) in _TIFF_SIGNATURES


def _native_chunks(
    tiff: tifffile.TiffFile,
    *,
    series: int,
    level: int,
) -> tuple[int, ...]:
    """Read native chunk sizes from a short-lived store without decoding pixels."""
    # Use the same explicit singleton policy as semantic metadata. Shaped TIFF
    # series retain extra singleton axes in some supported tifffile versions.
    store = tiff.aszarr(series=series, level=level, squeeze=True)
    try:
        array = zarr.open_array(store=store, mode="r")
        return tuple(array.chunks)
    finally:
        store.close()


def _sample_color(level: tifffile.TiffPageSeries, axes: tuple[str, ...], shape: tuple[int, ...]) -> SampleColor | None:
    """Return the colour model the first page of ``level`` declares for its samples axis ``S``."""
    if "S" not in axes:
        return None
    page = level.keyframe
    extras = tuple(page.extrasamples)
    color: SampleColor | None = None
    if page.photometric == tifffile.PHOTOMETRIC.RGB:
        if not extras:
            color = "rgb"
        elif len(extras) == 1 and extras[0] in _ALPHA_SAMPLES:
            color = "rgba"
    elif (
        page.photometric == tifffile.PHOTOMETRIC.YCBCR
        and page.compression in _JPEG_COMPRESSIONS
        and page.planarconfig == tifffile.PLANARCONFIG.CONTIG
        and not extras
    ):
        color = "rgb"
    # The declaration names the exposed samples axis only if it has one sample per letter.
    if color is None or shape[axes.index("S")] != len(color):
        return None
    return color


def _level_metadata(
    tiff: tifffile.TiffFile,
    level: tifffile.TiffPageSeries,
    *,
    series_index: int,
    level_index: int,
) -> TiffLevelMetadata:
    """Describe one pyramid level with consistent shape, axes, and native chunks."""
    if tiff.is_ome and any(page is None for page in level.pages):
        message = f"OME series {series_index} level {level_index} has a missing image plane."
        raise RasterFormatError(message)
    shape = tuple(level.shape)
    axes = tuple(level.axes)
    native_chunks = _native_chunks(tiff, series=series_index, level=level_index)
    if not shape or any(size <= 0 for size in shape):
        message = f"TIFF series {series_index} level {level_index} has invalid shape {shape}."
        raise RasterFormatError(message)
    # tifffile versions differ in whether series metadata retains shaped-file
    # singleton C/T/Z axes. Match squeeze=True in both metadata and pixel stores,
    # while retaining singleton spatial dimensions.
    dimensions = [(size, axis) for size, axis in zip(shape, axes, strict=True) if size != 1 or axis in ("X", "Y")]
    shape = tuple(size for size, _ in dimensions)
    axes = tuple(axis for _, axis in dimensions)
    if len(native_chunks) != len(shape) or any(size <= 0 for size in native_chunks):
        message = (
            f"TIFF series {series_index} level {level_index} has invalid native "
            f"chunks {native_chunks} for shape {shape}."
        )
        raise RasterFormatError(message)
    return TiffLevelMetadata(
        shape=shape,
        # Canonicalize equivalent NumPy aliases (notably longlong) for Zarr
        # interoperability; dtype.str preserves kind, itemsize, and byte order.
        dtype=np.dtype(level.dtype.str),
        axes=axes,
        native_chunks=native_chunks,
        color=_sample_color(level, axes, shape),
    )


def _parse_ome_xml(path: Path, ome_xml: str | None) -> OME | None:
    """Parse optional OME metadata and attach source context to malformed XML."""
    if ome_xml is None:
        return None
    try:
        return from_xml(ome_xml, validate=False)
    except Exception as error:
        message = f"Cannot parse OME metadata from TIFF source: {path}"
        raise RasterFormatError(message) from error


def _require_linked_tiff(path: Path) -> None:
    """Check each external OME source before tifffile can follow the reference."""
    if not path.is_file():
        message = f"OME metadata references a non-file TIFF source: {path}"
        raise RasterFormatError(message)
    if not _has_tiff_signature(path):
        message = f"Linked OME source does not have a TIFF signature: {path}"
        raise RasterFormatError(message)


def _remaining_planes(pixels: Pixels, block: TiffData) -> int:
    """Count logical planes from FirstZ/C/T in OME's fastest-to-slowest dimension order."""
    samples = (pixels.channels[0].samples_per_pixel or 1) if pixels.channels else 1
    dimensions = {
        "Z": (pixels.size_z, block.first_z),
        "C": (pixels.size_c // samples, block.first_c),
        "T": (pixels.size_t, block.first_t),
    }
    stride, offset = 1, 0
    for axis in pixels.dimension_order.value[2:]:
        extent, first = dimensions[axis]
        offset += first * stride
        stride *= extent
    return stride - offset


def _validate_ome_mappings(root: Path, tiff: tifffile.TiffFile, ome: OME) -> None:
    """Validate linked files before series discovery and compare IFD ranges per image."""
    root = root.resolve()
    checked_sources = {root}
    intervals: dict[Path, list[tuple[int, int, int]]] = {}

    @cache
    def file_plane_count(path: Path) -> int:
        if path == root:
            return len(tiff.pages)
        with tifffile.TiffFile(path) as other:
            return len(other.pages)

    for image_number, image in enumerate(ome.images):
        for block in image.pixels.tiff_data_blocks:
            linked = _tiff_data_path(root, ome, block)
            if linked not in checked_sources:
                _require_linked_tiff(linked)
                checked_sources.add(linked)
            count = block.plane_count
            if count == 0:
                # tifffile interprets zero as an omitted count and would expose undeclared pixels.
                message = f"OME image {image_number} declares PlaneCount=0 for {linked}."
                raise RasterFormatError(message)
            if len(ome.images) == 1:
                continue
            if count is None:
                count = 1 if "ifd" in block.model_fields_set else file_plane_count(linked)
            used = min(count, _remaining_planes(image.pixels, block))
            if used > 0:
                intervals.setdefault(linked, []).append((block.ifd, block.ifd + used, image_number))
    _reject_overlapping_ifds(intervals)


def _reject_overlapping_ifds(intervals: dict[Path, list[tuple[int, int, int]]]) -> None:
    """Reject a physical IFD claimed by distinct images, even when later TiffData overwrites a mapping."""
    for path, runs in intervals.items():
        previous_image, end = -1, 0
        for start, stop, image in sorted(runs):
            if start < end and image != previous_image:
                message = (
                    f"OME image {image} reuses TIFF IFD {start} in {path} already owned by image {previous_image}."
                )
                raise RasterFormatError(message)
            previous_image, end = image, max(stop, end)


def _tiff_data_path(root: Path, ome: OME, block: TiffData) -> Path:
    """Return the TIFF file that stores one OME ``TiffData`` block.

    A block without a file reference, or whose UUID is the document's own UUID,
    is stored in the inspected file. Renamed single-file OME-TIFFs keep their
    original ``FileName`` beside their own UUID, and tifffile resolves such
    references by UUID, as the OME-TIFF specification intends.
    """
    if block.uuid is None or block.uuid.file_name is None or block.uuid.value == ome.uuid:
        return root
    # tifffile opens linked files beside the source's real path, so a symlinked
    # source is checked where tifffile reads instead of zero-filling missing data.
    directory = root.parent
    linked = (directory / block.uuid.file_name).resolve()
    if not linked.is_relative_to(directory):
        message = (
            f"OME metadata reference escapes the TIFF source directory {directory}: "
            f"{block.uuid.file_name!r} resolves to {linked}."
        )
        raise RasterFormatError(message)
    return linked


def inspect_tiff(path: ArtifactFile, /, *, context: ReaderErrorContext) -> TiffMetadata:
    """Inspect all TIFF series and levels without decoding raster pixels.

    Parameters
    ----------
    path
        Validated regular file holding a classic TIFF, OME-TIFF, BigTIFF, or
        BTF source. Format bytes, rather than the file-name suffix, determine
        whether the source is a TIFF.
    context
        Reader and artifact identity added to inspection failures.

    Returns
    -------
    TiffMetadata
        A closed immutable description suitable for an explicit
        :func:`read_tiff` call.

    Raises
    ------
    RasterFormatError
        If the TIFF signature, OME image mapping, linked files, or storage
        metadata violates the supported contract.
    OSError
        If the file cannot be read.

    Notes
    -----
    Inspection opens short-lived Zarr stores only to obtain native chunk
    geometry. It never requests a pixel selection or decodes a TIFF segment.
    Nonspatial singleton axes are omitted explicitly; X and Y remain present
    even when their length is one, independently of TIFF metadata defaults.
    Failures keep their type and gain a note naming ``context`` and ``path``.
    """
    metadata = inspect_tiff_if_recognized(path, context=context)
    if metadata is None:
        message = f"File does not contain a classic TIFF or BigTIFF signature: {path}"
        error = RasterFormatError(message)
        add_reader_context(error, context=context, path=path)
        raise error
    return metadata


def inspect_tiff_if_recognized(path: ArtifactFile, /, *, context: ReaderErrorContext) -> TiffMetadata | None:
    """Inspect a TIFF by content. Return None for other signatures and propagate failures of recognized TIFFs."""
    try:
        return _inspect(path) if _has_tiff_signature(path) else None
    except (OSError, ValueError) as error:
        add_reader_context(error, context=context, path=path)
        raise


def _inspect(path: ArtifactFile) -> TiffMetadata:
    """Open the TIFF once and describe every series and level without reading pixels."""
    with tifffile.TiffFile(path) as tiff:
        ome = _parse_ome_xml(path, tiff.ome_metadata)
        if ome is not None:
            _validate_ome_mappings(path, tiff, ome)
            if len(ome.images) != len(tiff.series):
                message = (
                    f"OME metadata declares {len(ome.images)} images, "
                    f"but tifffile discovered {len(tiff.series)} image series."
                )
                raise RasterFormatError(message)
        series_metadata: list[TiffSeriesMetadata] = []
        for series_index, series in enumerate(tiff.series):
            levels = tuple(
                _level_metadata(
                    tiff,
                    level,
                    series_index=series_index,
                    level_index=level_index,
                )
                for level_index, level in enumerate(series.levels)
            )
            series_metadata.append(TiffSeriesMetadata(levels=levels))
    return TiffMetadata(
        path=path,
        series=tuple(series_metadata),
        ome=ome,
    )
