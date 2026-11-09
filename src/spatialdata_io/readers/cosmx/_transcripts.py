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
``global`` (scale to micrometres) with it, and every FOV's local pixel frame
``fov_<n>_pixels`` through the inverse of its placement.
"""

from __future__ import annotations

import warnings
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from dask import delayed
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity, Scale

from spatialdata_io.readers._utils.csv import csv_errors, read_csv_blocks
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers.cosmx._schemas import (
    CELL,
    CELL_ID,
    FOV,
    TRANSCRIPTS,
    require_finite,
    require_positive,
    schema_error,
    transcript_types,
)
from spatialdata_io.readers.cosmx._tables import read_header
from spatialdata_io.readers.cosmx._transforms import fov_frame, global_frame

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dask.delayed import Delayed
    from spatialdata.transformations import BaseTransformation

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.cosmx._transforms import Placement

_GZIP_RANGE_BYTES = 8 * 1024 * 1024
_READ_BYTES = 1024 * 1024


def _normalize(table: pa.Table, *, features: tuple[str, ...], path: ArtifactFile) -> pd.DataFrame:
    """Validate one typed partition and convert only required point-model columns."""
    require_positive(table[FOV], path, TRANSCRIPTS, FOV)
    for name in ("x_global_px", "y_global_px", "x_local_px", "y_local_px"):
        require_finite(table[name], path, TRANSCRIPTS, name)
    names = ["x_global_px", "y_global_px", "target", FOV, CELL_ID, "x_local_px", "y_local_px"]
    names.extend(name for name in ("z", "CellComp", CELL) if name in table.column_names)
    frame = table.select(names).to_pandas(types_mapper=pd.ArrowDtype)
    vocabulary = pd.Index(features)
    codes = vocabulary.get_indexer(pd.Index(frame["target"].array))
    if (codes < 0).any():
        raise schema_error(
            path, TRANSCRIPTS, found="missing or unknown targets", expected="targets from the count vocabulary"
        )
    frame["target"] = pd.Categorical.from_codes(codes, categories=vocabulary)
    # PointsModel requires NumPy coordinates; local coordinates stay Arrow annotations.
    frame["x_global_px"] = frame["x_global_px"].to_numpy(dtype="float64")
    frame["y_global_px"] = frame["y_global_px"].to_numpy(dtype="float64")
    return frame.rename(
        columns={"x_global_px": "x", "y_global_px": "y", "x_local_px": "x_local", "y_local_px": "y_local", "z": "z_raw"}
    )


@dataclass(frozen=True, slots=True)
class _Projection:
    """Immutable task state: the source header, projected fields, and target vocabulary."""

    names: tuple[str, ...]
    types: pa.Schema
    features: tuple[str, ...]
    path: ArtifactFile


def _parse(payload: bytes | bytearray, projection: _Projection) -> pd.DataFrame:
    """Parse gzip records with the same physical types and null policy as plain blocks."""
    path = projection.path
    if not payload:
        return _normalize(projection.types.empty_table(), features=projection.features, path=path)
    with pa.BufferReader(payload) as source, csv_errors(context=ReaderErrorContext("cosmx", TRANSCRIPTS), path=path):
        table = pacsv.read_csv(
            source,
            read_options=pacsv.ReadOptions(column_names=projection.names, use_threads=False),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(
                include_columns=projection.types.names,
                column_types=projection.types,
                null_values=[],
                strings_can_be_null=False,
            ),
        )
    return _normalize(table, features=projection.features, path=path)


def _normalize_block(block: pd.DataFrame, projection: _Projection) -> pd.DataFrame:
    """Validate one parsed plain-CSV block."""
    # Arrow-backed columns convert without copying; a thread pool would only add start-up time.
    table = pa.Table.from_pandas(block, preserve_index=False, nthreads=1)
    return _normalize(table, features=projection.features, path=projection.path)


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


def _gzip_range(projection: _Projection, offset: int, *, start: int, stop: int) -> tuple[int, pd.DataFrame]:
    """Read contiguously from the predecessor's verified boundary, never resynchronizing.

    The scalar end offset orders source reads without keeping open resources in
    the graph. A member crossing several ranges belongs wholly to its first task;
    those subsequent ranges return empty partitions.
    """
    payload = bytearray()
    pending = b""
    with projection.path.open("rb") as handle:
        while offset < stop:
            offset, pending = _decode_member(handle, offset, projection.path, payload, pending)
    if start == 0:
        del payload[: payload.find(b"\n") + 1]
    return offset, _parse(payload, projection)


def _frame(projection: _Projection, meta: pd.DataFrame) -> dd.DataFrame:
    """Return one partition per plain-CSV block or compressed member range."""
    path = projection.path
    if not path.name.endswith(".gz"):
        context = ReaderErrorContext("cosmx", TRANSCRIPTS)
        blocks = read_csv_blocks(
            path,
            header=projection.names,
            columns=projection.types.names,
            column_types={field.name: field.type for field in projection.types},
            context=context,
        )
        return blocks.map_partitions(_normalize_block, projection, meta=meta)

    size = path.stat().st_size
    offset: int | Delayed = 0
    partitions: list[Delayed] = []
    for start in range(0, size, _GZIP_RANGE_BYTES):
        task = delayed(_gzip_range, nout=2)(projection, offset, start=start, stop=min(start + _GZIP_RANGE_BYTES, size))
        offset, partition = task
        partitions.append(partition)
    return dd.from_delayed(partitions, meta=meta)


def read_transcripts(
    path: ArtifactFile,
    *,
    features: tuple[str, ...],
    placements: Mapping[int, Placement],
    pixel_size_um: float | None = None,
) -> dd.DataFrame:
    """Return lazy global XY points and every FOV frame after header inspection.

    Records must each occupy one physical line. Values are validated when their
    partition is computed.
    """
    names = read_header(path, TRANSCRIPTS)
    types = transcript_types(names, path)
    meta = _normalize(types.empty_table(), features=features, path=path)
    frame = _frame(_Projection(names, types, features, path), meta)

    transformations: dict[str, BaseTransformation] = {
        global_frame(pixel_size_um): (
            Identity() if pixel_size_um is None else Scale([pixel_size_um, pixel_size_um], axes=("x", "y"))
        )
    }
    transformations.update({fov_frame(fov): placement.global_to_local() for fov, placement in placements.items()})
    # Partitions are deliberately unsorted. Known target categories and
    # sort=False avoid the public model's eager category and index scans.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The index of the dataframe is not monotonic increasing.*", category=UserWarning
        )
        return PointsModel.parse(
            frame, coordinates={"x": "x", "y": "y"}, feature_key="target", transformations=transformations, sort=False
        )
