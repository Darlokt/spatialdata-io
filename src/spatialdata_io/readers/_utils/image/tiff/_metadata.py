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

Readers that assign meaning to physical pages can request the ordered
top-level IFD tags, including the original resolution and position rationals.
ImageJ plane labels can be requested independently. Inspection closes the
TIFF and its temporary stores before returning either selection.
"""

from __future__ import annotations

from functools import cache
from operator import index, itemgetter
from typing import TYPE_CHECKING

import numpy as np
import tifffile
import zarr
from ome_types import from_xml

from spatialdata_io.readers._utils.errors import RasterFormatError, add_reader_context
from spatialdata_io.readers._utils.image.tiff._rows import direct_row_axis, physical_page
from spatialdata_io.readers._utils.image.tiff._types import (
    TiffLevelMetadata,
    TiffMetadata,
    TiffPageMetadata,
    TiffSeriesMetadata,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ome_types import OME
    from ome_types.model import Pixels, TiffData
    from tifffile.zarr import ZarrTiffStore

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.image._types import SampleColor
    from spatialdata_io.readers._utils.image.tiff._rows import TiffImage
    from spatialdata_io.readers._utils.paths import ArtifactFile

_TIFF_SIGNATURES = frozenset({b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"})
# Old-style, new-style, lossy (DNG) and Aperio JPEG: the compressions whose
# YCbCr data tifffile decodes to RGB.
_JPEG_COMPRESSIONS = frozenset({6, 7, 33007, 34892})
_ALPHA_SAMPLES = frozenset({tifffile.EXTRASAMPLE.ASSOCALPHA, tifffile.EXTRASAMPLE.UNASSALPHA})
_MAX_PAGE_TEXT_BYTES = 2**20
_MAX_IMAGEJ_METADATA_BYTES = 16 * 2**20


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
    return _store_chunks(tiff.aszarr(series=series, level=level, squeeze=True))


def _store_chunks(store: ZarrTiffStore) -> tuple[int, ...]:
    try:
        array = zarr.open_array(store=store, mode="r")
        return tuple(array.chunks)
    finally:
        store.close()


def _sample_color(image: TiffImage, axes: tuple[str, ...], shape: tuple[int, ...]) -> SampleColor | None:
    """Return the colour model the first page of ``image`` declares for its samples axis ``S``."""
    if "S" not in axes:
        return None
    page = image.keyframe
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
        direct_row_axis=direct_row_axis(tiff, level, shape),
    )


def _page_text(page: tifffile.TiffPage, code: int) -> str:
    tag = page.tags.get(code)
    if tag is None:
        return ""
    if tag.count <= _MAX_PAGE_TEXT_BYTES and isinstance(value := tag.value, str):
        return value
    message = f"TIFF IFD {page.index} needs {tag.name} as text of at most {_MAX_PAGE_TEXT_BYTES} bytes."
    raise RasterFormatError(message)


def _rational_tag(page: tifffile.TiffPage, code: int) -> tuple[int, int] | None:
    tag = page.tags.get(code)
    if tag is None:
        return None
    value: object = tag.value
    match value:
        case (int() as numerator, int() as denominator):
            return numerator, denominator
    message = f"TIFF IFD {page.index} needs a rational {tag.name} tag."
    raise RasterFormatError(message)


def _integer_tag(page: tifffile.TiffPage, code: int) -> int | None:
    tag = page.tags.get(code)
    if tag is None:
        return None
    value: object = tag.value
    if isinstance(value, int):
        return index(value)
    message = f"TIFF IFD {page.index} needs an integer {tag.name} tag."
    raise RasterFormatError(message)


def _sample_tag(page: tifffile.TiffPage, code: int) -> int | float | tuple[int | float, ...] | None:
    tag = page.tags.get(code)
    if tag is None:
        return None
    value: object = tag.value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, bytes):
        return value[0] if len(value) == 1 else tuple(value)
    if isinstance(value, tuple) and all(isinstance(item, (int, float)) for item in value):
        return value
    message = f"TIFF IFD {page.index} needs numeric {tag.name} values."
    raise RasterFormatError(message)


def _page_metadata(tiff: tifffile.TiffFile, page: tifffile.TiffPage) -> TiffPageMetadata:
    shape, axes, dtype = page.shape, tuple(page.axes), page.dtype
    if not shape or any(size <= 0 for size in shape) or len(axes) != len(shape) or dtype is None:
        message = f"TIFF IFD {page.index} has invalid raster metadata (shape {shape}, axes {axes}, dtype {dtype})."
        raise RasterFormatError(message)
    native_chunks = _store_chunks(page.aszarr())
    if len(native_chunks) != len(shape) or any(size <= 0 for size in native_chunks):
        message = f"TIFF IFD {page.index} has invalid native chunks {native_chunks} for shape {shape}."
        raise RasterFormatError(message)
    return TiffPageMetadata(
        shape=shape,
        dtype=np.dtype(dtype.str),
        axes=axes,
        native_chunks=native_chunks,
        color=_sample_color(page, axes, shape),
        direct_row_axis=direct_row_axis(tiff, page, shape),
        index=page.index,
        description=_page_text(page, 270),
        software=_page_text(page, 305),
        x_resolution=_rational_tag(page, 282),
        y_resolution=_rational_tag(page, 283),
        resolution_unit=_integer_tag(page, 296),
        x_position=_rational_tag(page, 286),
        y_position=_rational_tag(page, 287),
        orientation=_integer_tag(page, 274),
        photometric=index(page.photometric),
        samples_per_pixel=page.samplesperpixel,
        min_sample_value=_sample_tag(page, 340),
        max_sample_value=_sample_tag(page, 341),
    )


def _imagej_labels(tiff: tifffile.TiffFile) -> tuple[str, ...] | None:
    if not tiff.is_imagej or (tag := tiff.pages.first.tags.get(50839)) is None:
        return None
    # tifffile decodes IJMetadata when it reads the IFD, including overlays
    # and LUTs. This limit applies to the tag accepted for label inspection.
    metadata: object = tag.value
    if tag.count > _MAX_IMAGEJ_METADATA_BYTES or not isinstance(metadata, dict):
        message = f"ImageJ Labels need a decoded IJMetadata tag of at most {_MAX_IMAGEJ_METADATA_BYTES} bytes."
        raise RasterFormatError(message)
    labels: object = metadata.get("Labels")
    if labels is None:
        return None
    if isinstance(labels, str):
        return (labels,)
    if isinstance(labels, (list, tuple)) and all(isinstance(label, str) for label in labels):
        return tuple(labels)
    message = "ImageJ Labels must contain text entries."
    raise RasterFormatError(message)


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


def _logical_plane_range(pixels: Pixels, block: TiffData) -> tuple[int, int]:
    """Return the first logical plane and total planes in OME dimension order."""
    samples = (pixels.channels[0].samples_per_pixel or 1) if pixels.channels else 1
    dimensions = {
        "Z": (pixels.size_z, block.first_z),
        "C": (pixels.size_c // samples, block.first_c),
        "T": (pixels.size_t, block.first_t),
    }
    stride, offset = 1, 0
    for axis in pixels.dimension_order.value[2:]:
        extent, first = dimensions[axis]
        if first >= extent:
            message = f"OME First{axis}={first} is outside the {extent} logical {axis} planes in {pixels.id}."
            raise RasterFormatError(message)
        offset += first * stride
        stride *= extent
    return offset, stride


def _validate_ome_mappings(root: Path, tiff: tifffile.TiffFile, ome: OME) -> None:
    """Validate linked files before series discovery and compare IFD ranges per image."""
    root = root.resolve()
    paths_by_uuid: dict[str, Path] = {}
    physical: dict[Path, list[tuple[int, int, tuple[int, int]]]] = {}
    logical: dict[int, list[tuple[int, int, tuple[Path, int]]]] = {}

    @cache
    def file_plane_count(path: Path) -> int:
        if path == root:
            return len(tiff.pages)
        _require_linked_tiff(path)
        with tifffile.TiffFile(path) as other:
            return len(other.pages)

    for image_number, image in enumerate(ome.images):
        for block in image.pixels.tiff_data_blocks:
            linked = _tiff_data_path(root, ome, block, paths_by_uuid)
            count = block.plane_count
            if count == 0:
                # tifffile interprets zero as an omitted count and would expose undeclared pixels.
                message = f"OME image {image_number} declares PlaneCount=0 for {linked}."
                raise RasterFormatError(message)
            start, total = _logical_plane_range(image.pixels, block)
            file_planes = file_plane_count(linked)
            if count is None:
                count = 1 if "ifd" in block.model_fields_set else file_planes
            used = min(count, total - start)
            if block.ifd + used > file_planes:
                message = (
                    f"OME image {image_number} needs IFD {block.ifd + used - 1} in {linked}, "
                    f"with only {file_planes} IFDs."
                )
                raise RasterFormatError(message)
            if len(ome.images) == 1 and len(image.pixels.tiff_data_blocks) == 1:
                continue
            # Within a block the IFD and the logical plane increase together.
            # Equal offsets and owners give identical assignments over an overlap.
            delta = start - block.ifd
            physical.setdefault(linked, []).append((block.ifd, block.ifd + used, (image_number, delta)))
            logical.setdefault(image_number, []).append((start, start + used, (linked, delta)))

    _require_unique_planes(physical, logical)


def _require_unique_planes(
    physical: dict[Path, list[tuple[int, int, tuple[int, int]]]],
    logical: dict[int, list[tuple[int, int, tuple[Path, int]]]],
) -> None:
    """Require both directions of the OME image-plane to file-IFD mapping to be unambiguous."""
    for path, runs in physical.items():
        if (ifd := _first_conflict(runs)) is not None:
            message = f"OME mapping reuses TIFF IFD {ifd} in {path} for different image planes."
            raise RasterFormatError(message)
    for number, plane_runs in logical.items():
        if (plane := _first_conflict(plane_runs)) is not None:
            message = f"OME image {number} assigns logical plane {plane} to different TIFF IFDs."
            raise RasterFormatError(message)


def _first_conflict[Owner](runs: list[tuple[int, int, Owner]]) -> int | None:
    """Locate the first overlapping half-open interval with a different plane mapping."""
    end = 0
    previous: Owner | None = None
    runs.sort(key=itemgetter(0))
    for start, stop, owner in runs:
        if start < end and owner != previous:
            return start
        previous, end = owner, max(stop, end)
    return None


def _tiff_data_path(root: Path, ome: OME, block: TiffData, paths_by_uuid: dict[str, Path]) -> Path:
    """Return the TIFF file that stores one OME ``TiffData`` block.

    A block without a file reference, or whose UUID is the document's own UUID,
    is stored in the inspected file. Renamed single-file OME-TIFFs keep their
    original ``FileName`` beside their own UUID, and tifffile resolves such
    references by UUID, as the OME-TIFF specification intends.
    """
    if block.uuid is None or block.uuid.value == ome.uuid:
        return root
    uuid = block.uuid.value
    known = paths_by_uuid.get(uuid)
    if block.uuid.file_name is None:
        if known is None:
            message = f"OME reference to UUID {uuid} needs a FileName on its first external reference."
            raise RasterFormatError(message)
        return known
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
    if known is not None and linked != known:
        message = f"OME UUID {uuid} has conflicting file names resolving to {known} and {linked}."
        raise RasterFormatError(message)
    paths_by_uuid[uuid] = linked
    return linked


def inspect_tiff(
    path: ArtifactFile,
    /,
    *,
    context: ReaderErrorContext,
    include_pages: bool = False,
    include_imagej_labels: bool = False,
) -> TiffMetadata:
    """Inspect all TIFF series and levels without decoding raster pixels.

    Parameters
    ----------
    path
        Validated regular file holding a classic TIFF, OME-TIFF, BigTIFF, or
        BTF source. Format bytes, rather than the file-name suffix, determine
        whether the source is a TIFF.
    context
        Reader and artifact identity added to inspection failures.
    include_pages
        Inspect physical top-level IFDs and their selected standard tags in
        file order. Their indices select individual pages in :func:`read_tiff`.
    include_imagej_labels
        Preserve optional ImageJ Labels from an IJMetadata tag of at most
        16 MiB as a tuple in storage order.

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
    TIFF page parsing and series discovery can decode metadata in tifffile
    before the opt-in page and ImageJ tag limits are applied.
    Nonspatial singleton axes are omitted explicitly; X and Y remain present
    even when their length is one, independently of TIFF metadata defaults.
    Failures keep their type and gain a note naming ``context`` and ``path``.
    """
    metadata = inspect_tiff_if_recognized(
        path, context=context, include_pages=include_pages, include_imagej_labels=include_imagej_labels
    )
    if metadata is None:
        message = f"File does not contain a classic TIFF or BigTIFF signature: {path}"
        error = RasterFormatError(message)
        add_reader_context(error, context=context, path=path)
        raise error
    return metadata


def inspect_tiff_if_recognized(
    path: ArtifactFile,
    /,
    *,
    context: ReaderErrorContext,
    include_pages: bool = False,
    include_imagej_labels: bool = False,
) -> TiffMetadata | None:
    """Inspect a TIFF by content. Return None for other signatures and propagate failures of recognized TIFFs."""
    try:
        if not _has_tiff_signature(path):
            return None
        return _inspect(path, include_pages=include_pages, include_imagej_labels=include_imagej_labels)
    except (OSError, ValueError) as error:
        add_reader_context(error, context=context, path=path)
        raise


def _inspect(path: ArtifactFile, *, include_pages: bool, include_imagej_labels: bool) -> TiffMetadata:
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
        pages = tuple(_page_metadata(tiff, physical_page(page)) for page in tiff.pages) if include_pages else None
        imagej_labels = _imagej_labels(tiff) if include_imagej_labels else None
    return TiffMetadata(
        path=path,
        series=tuple(series_metadata),
        ome=ome,
        pages=pages,
        imagej_labels=imagej_labels,
    )
