"""Inspect TIFF storage and semantic metadata without decoding pixels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tifffile
import zarr
from ome_types import from_xml

from spatialdata_io.readers._utils.errors import RasterFormatError
from spatialdata_io.readers._utils.image._path import normalize_file_path
from spatialdata_io.readers._utils.image.tiff._types import (
    TiffLevelMetadata,
    TiffMetadata,
    TiffSeriesMetadata,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ome_types import OME
    from ome_types.model import TiffData

_TIFF_SIGNATURES = frozenset({b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"})


def _validate_tiff_signature(path: Path) -> None:
    """Reject non-TIFF bytes before opening a metadata or pixel decoder."""
    with path.open("rb") as stream:
        signature = stream.read(4)
    if signature not in _TIFF_SIGNATURES:
        message = f"File does not contain a classic TIFF or BigTIFF signature: {path}"
        raise RasterFormatError(message)


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


def _level_metadata(
    tiff: tifffile.TiffFile,
    level: tifffile.TiffPageSeries,
    *,
    series_index: int,
    level_index: int,
) -> TiffLevelMetadata:
    """Describe one pyramid level with consistent shape, axes, and native chunks."""
    shape = tuple(level.shape)
    axes = tuple(level.axes)
    native_chunks = _native_chunks(tiff, series=series_index, level=level_index)
    if not shape or any(size <= 0 for size in shape):
        message = f"TIFF series {series_index} level {level_index} has invalid shape {shape}."
        raise RasterFormatError(message)
    if len(axes) != len(shape) or any(len(axis) != 1 for axis in axes):
        message = f"TIFF series {series_index} level {level_index} axes {axes!r} do not describe shape {shape}."
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


def _validate_ome_series(ome: OME | None, series_count: int) -> None:
    """Require every declared OME image to correspond to one discovered series."""
    if ome is None:
        return
    if len(ome.images) != series_count:
        message = (
            f"OME metadata declares {len(ome.images)} images, but tifffile discovered {series_count} image series."
        )
        raise RasterFormatError(message)


def _validate_ome_image_mappings(ome: OME | None, root: Path) -> None:
    """Reject OME images that claim the same file and starting TIFF directory."""
    if ome is None:
        return
    mapped_by_previous_images: set[tuple[Path, int]] = set()
    for image_index, image in enumerate(ome.images):
        image_mappings = {
            (_tiff_data_path(root, ome, block), block.ifd or 0) for block in image.pixels.tiff_data_blocks
        }
        duplicates = image_mappings.intersection(mapped_by_previous_images)
        if duplicates:
            message = (
                f"OME image {image_index} reuses TIFF IFD mappings already owned by another "
                f"image: {sorted((str(path), ifd) for path, ifd in duplicates)}."
            )
            raise RasterFormatError(message)
        mapped_by_previous_images.update(image_mappings)


def _tiff_data_path(root: Path, ome: OME, block: TiffData) -> Path:
    """Return the TIFF file that stores one OME ``TiffData`` block.

    A block without a file reference, or whose UUID is the document's own UUID,
    is stored in the inspected file. Renamed single-file OME-TIFFs keep their
    original ``FileName`` beside their own UUID, and tifffile resolves such
    references by UUID, as the OME-TIFF specification intends.
    """
    if block.uuid is None or block.uuid.file_name is None or block.uuid.value == ome.uuid:
        return root
    return _resolve_linked_path(root, block.uuid.file_name)


def _resolve_linked_path(root: Path, file_name: str) -> Path:
    """Resolve an OME file reference without escaping the source directory."""
    linked = (root.parent / file_name).resolve()
    if not linked.is_relative_to(root.parent):
        message = (
            f"OME metadata reference escapes the TIFF source directory {root.parent}: "
            f"{file_name!r} resolves to {linked}."
        )
        raise RasterFormatError(message)
    return linked


def _validate_linked_paths(root: Path, ome: OME | None) -> None:
    """Require metadata-linked TIFF sources to exist within the dataset boundary."""
    if ome is not None:
        for image in ome.images:
            for tiff_data in image.pixels.tiff_data_blocks:
                linked = _tiff_data_path(root, ome, tiff_data)
                if linked == root:
                    continue
                if not linked.is_file():
                    message = f"OME metadata references a non-file TIFF source: {linked}"
                    raise RasterFormatError(message)
                _validate_tiff_signature(linked)


def inspect_tiff(
    path: str | Path,
) -> TiffMetadata:
    """Inspect all TIFF series and levels without decoding raster pixels.

    Parameters
    ----------
    path
        Classic TIFF, OME-TIFF, BigTIFF, or BTF source. Format bytes, rather
        than the file-name suffix, determine whether the source is a TIFF.

    Returns
    -------
    TiffMetadata
        A closed immutable description suitable for an explicit
        :func:`read_tiff` call.

    Raises
    ------
    RasterFormatError
        If the path, TIFF signature, OME image mapping, linked files, or storage
        metadata violates the supported contract.

    Notes
    -----
    Inspection opens short-lived Zarr stores only to obtain native chunk
    geometry. It never requests a pixel selection or decodes a TIFF segment.
    Nonspatial singleton axes are omitted explicitly; X and Y remain present
    even when their length is one, independently of TIFF metadata defaults.
    """
    normalized_path = normalize_file_path(path)
    _validate_tiff_signature(normalized_path)

    with tifffile.TiffFile(normalized_path) as tiff:
        ome_xml = tiff.ome_metadata
        ome = _parse_ome_xml(normalized_path, ome_xml)
        _validate_linked_paths(normalized_path, ome)
        _validate_ome_series(ome, len(tiff.series))
        _validate_ome_image_mappings(ome, normalized_path)
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
        path=normalized_path,
        series=tuple(series_metadata),
        ome=ome,
    )
