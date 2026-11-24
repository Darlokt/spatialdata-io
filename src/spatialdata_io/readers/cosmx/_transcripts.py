"""Construct one projected, path-backed CosMx transcript points element.

``<prefix>_tx_file.csv`` holds one record per decoded transcript, each on one
physical line. Points are placed at the global pixel position
(``x_global_px``, ``y_global_px``); the local position, ``fov``, ``cell_ID``
(0 for unassigned transcripts), ``target`` and the optional ``z``, ``CellComp``
and ``cell`` fields are annotations. Construction reads only the header; every
partition is parsed and validated when computed.

Plain CSV is split by the shared CSV block reader into newline-delimited byte
blocks of about 8 MiB, its default.

Gzip sources (``.csv.gz``) are read member by member. AtoMx writes
concatenated, independently compressed members, each ending at a record
boundary; the first may hold only the header or the header and rows::

    byte 0                                                            EOF
    | member: header [+ rows] | member: rows | member: rows | ... |
    |<----- range 0 (8 MiB) ----->|<----- range 1 ----->|<- range 2 ->|

One task owns the members that start in its compressed range, including the
whole member crossing its end, so a range wholly inside a member yields an
empty partition. Each task continues from the exact end offset that its
predecessor verified, never scanning for a gzip magic number, and checks every
member's deflate stream, CRC and size trailer; corrupt or trailing data fails.
Source decoding is therefore sequential. A single-stream gzip file is one
member and forms one nonempty partition with all its rows.

Coordinate systems: ``global_pixels`` (identity) without ``pixel_size_um``, or
``global`` (scale to micrometres) with it, and every placed FOV's local pixel frame
``fov_<n>_pixels`` through the inverse of its placement.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO

import dask.dataframe as dd
import pandas as pd
from dask import delayed
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity, Scale

from spatialdata_io.readers._utils.csv import csv_header, parse_csv_records, read_csv_blocks
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.points import suppress_points_index_warning
from spatialdata_io.readers.cosmx._schemas import (
    TRANSCRIPTS,
    normalize_transcript_partition,
    schema_error,
    transcript_types,
)
from spatialdata_io.readers.cosmx._transforms import fov_frame, global_frame

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pyarrow as pa
    from dask.delayed import Delayed
    from spatialdata.transformations import BaseTransformation

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.cosmx._layout import CsvFile
    from spatialdata_io.readers.cosmx._transforms import Placement

_GZIP_RANGE_BYTES = 8 * 1024 * 1024
_READ_BYTES = 1024 * 1024
_CONTEXT = ReaderErrorContext("cosmx", TRANSCRIPTS)


@dataclass(frozen=True, slots=True)
class _Projection:
    """Immutable task state: the source header, projected fields, and target vocabulary."""

    names: tuple[str, ...]
    types: pa.Schema
    features: pd.Index
    source: CsvFile


def _decode_member(
    handle: BinaryIO, offset: int, path: ArtifactFile, payload: bytearray, pending: bytes
) -> tuple[int, bytes]:
    """Append one member from an exact boundary, verifying its CRC and trailer."""
    # Reuse bounded compressed read-ahead across members within the same owned task.
    handle.seek(offset + len(pending))
    decoder = zlib.decompressobj(31)
    member_start = len(payload)
    while not decoder.eof:
        chunk = pending or handle.read(_READ_BYTES)
        if not chunk:
            raise schema_error(
                path, TRANSCRIPTS, found=f"corrupt gzip member at byte {offset}", expected="complete members"
            )
        try:
            decoded = decoder.decompress(chunk, _READ_BYTES)
        except zlib.error as error:
            raise schema_error(
                path, TRANSCRIPTS, found=f"corrupt gzip member at byte {offset}: {error}", expected="complete members"
            ) from error
        payload.extend(decoded)
        pending = decoder.unconsumed_tail
    if len(payload) == member_start or not payload.endswith(b"\n"):
        raise schema_error(
            path,
            TRANSCRIPTS,
            found=f"gzip member at byte {offset} ending inside a record",
            expected="record-aligned members",
        )
    return handle.tell() - len(decoder.unused_data), decoder.unused_data


def _gzip_range(projection: _Projection, cursor: tuple[int, int], *, stop: int) -> tuple[tuple[int, int], pd.DataFrame]:
    """Read contiguously from the predecessor's verified boundary, never resynchronizing.

    The cursor carries the next compressed offset and the next transcript row.
    A member crossing several ranges belongs wholly to its first task;
    those subsequent ranges return empty partitions.
    """
    offset, first_row = cursor
    payload = bytearray()
    pending = b""
    path = projection.source.path
    with path.open("rb") as handle:
        while offset < stop:
            offset, pending = _decode_member(handle, offset, path, payload, pending)
    if cursor[0] == 0:
        del payload[: payload.find(b"\n") + 1]
    block = parse_csv_records(payload, header=projection.names, schema=projection.types, context=_CONTEXT, path=path)
    next_row = first_row + len(block)
    block.index = pd.RangeIndex(first_row, next_row)
    return (offset, next_row), normalize_transcript_partition(block, features=projection.features, path=path)


def _frame(projection: _Projection, meta: pd.DataFrame) -> dd.DataFrame:
    """Return one partition per plain-CSV block or compressed member range."""
    path = projection.source.path
    if not projection.source.compressed:
        blocks = read_csv_blocks(
            path,
            header=projection.names,
            schema=projection.types,
            context=_CONTEXT,
        )
        return blocks.map_partitions(normalize_transcript_partition, features=projection.features, path=path, meta=meta)

    size = path.stat().st_size
    cursor: tuple[int, int] | Delayed = (0, 0)
    partitions: list[Delayed] = []
    for start in range(0, size, _GZIP_RANGE_BYTES):
        task = delayed(_gzip_range, nout=2)(projection, cursor, stop=min(start + _GZIP_RANGE_BYTES, size))
        cursor, partition = task
        partitions.append(partition)
    return dd.from_delayed(partitions, meta=meta)


def read_transcripts(
    source: CsvFile,
    *,
    features: tuple[str, ...],
    placements: Mapping[int, Placement],
    pixel_size_um: float | None = None,
) -> dd.DataFrame:
    """Return lazy global XY points with local frames for placed FOVs.

    Records must each occupy one physical line. Values are validated when their
    partition is computed.
    """
    path = source.path
    names = csv_header(path, context=_CONTEXT, compressed=source.compressed)
    types = transcript_types(names, path)
    vocabulary = pd.Index(features)
    empty = types.empty_table().to_pandas(types_mapper=pd.ArrowDtype)
    meta = normalize_transcript_partition(empty, features=vocabulary, path=path)
    frame = _frame(_Projection(names, types, vocabulary, source), meta)

    transformations: dict[str, BaseTransformation] = {
        global_frame(pixel_size_um): (
            Identity() if pixel_size_um is None else Scale([pixel_size_um, pixel_size_um], axes=("x", "y"))
        )
    }
    transformations.update({fov_frame(fov): placement.global_to_local() for fov, placement in placements.items()})
    # Partitions are deliberately unsorted. Known target categories and
    # sort=False avoid the public model's eager category and index scans.
    with suppress_points_index_warning():
        return PointsModel.parse(
            frame, coordinates={"x": "x", "y": "y"}, feature_key="target", transformations=transformations, sort=False
        )
