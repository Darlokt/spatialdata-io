"""Construct one projected, path-backed CosMx transcript points element.

``<prefix>_tx_file.csv`` holds one record per decoded transcript, each on one
physical line. Points are placed at the global pixel position
(``x_global_px``, ``y_global_px``); the local position, ``fov``, ``cell_ID``
(0 for unassigned transcripts), ``target`` and the optional ``z``, ``CellComp``
and ``cell`` fields are annotations. Construction reads only the header; every
partition is parsed and validated when computed.

Plain CSV is split by Dask into newline-delimited byte blocks of about 8 MiB.

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

import glob
import warnings
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO, cast

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from dask import delayed
from dask.bytes import read_bytes
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity, Scale

from spatialdata_io.readers.cosmx._schemas import (
    CELL,
    CELL_ID,
    FOV,
    INTEGER_DTYPE,
    STRING_DTYPE,
    TRANSCRIPTS,
    float_values,
    integer_values,
    schema_error,
    transcript_columns,
)
from spatialdata_io.readers.cosmx._tables import read_header
from spatialdata_io.readers.cosmx._transforms import fov_frame, global_frame

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import numpy as np
    import numpy.typing as npt
    from dask.delayed import Delayed
    from spatialdata.transformations import BaseTransformation

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.cosmx._transforms import Placement

_BLOCK_BYTES = 8 * 1024 * 1024
_GZIP_RANGE_BYTES = 8 * 1024 * 1024
_READ_BYTES = 1024 * 1024


def _normalize(table: pa.Table, *, features: tuple[str, ...], path: ArtifactFile) -> pd.DataFrame:
    """Validate one lexical partition and convert it to the returned point schema."""
    target = pd.Series(table.column("target").to_pandas(types_mapper=pd.ArrowDtype), dtype=STRING_DTYPE)
    if target.isna().any() or not target.isin(features).all():
        raise schema_error(
            path, TRANSCRIPTS, found="missing or unknown targets", expected="targets from the count vocabulary"
        )

    def floats(name: str) -> npt.NDArray[np.float64]:
        return float_values(table.column(name), path, TRANSCRIPTS, name)

    def integers(name: str, minimum: int | None) -> pd.Series:
        return pd.Series(
            integer_values(table.column(name), path, TRANSCRIPTS, name, minimum=minimum), dtype=INTEGER_DTYPE
        )

    frame = pd.DataFrame(
        {
            "x": floats("x_global_px"),
            "y": floats("y_global_px"),
            "target": pd.Categorical(target.to_numpy(), categories=features),
            FOV: integers(FOV, 1),
            CELL_ID: integers(CELL_ID, 0),
            "x_local": floats("x_local_px"),
            "y_local": floats("y_local_px"),
        }
    )
    if "z" in table.column_names:
        # The vendor defines z as a slice number in arbitrary units; it can be negative.
        frame["z_raw"] = integers("z", None)
    for name in ("CellComp", CELL):
        if name in table.column_names:
            frame[name] = pd.Series(table.column(name).to_pandas(types_mapper=pd.ArrowDtype), dtype=STRING_DTYPE)
    return frame


@dataclass(frozen=True, slots=True)
class _Projection:
    """Immutable task state: the source header, projected fields, and target vocabulary."""

    names: tuple[str, ...]
    columns: tuple[str, ...]
    features: tuple[str, ...]
    path: ArtifactFile


def _parse(payload: bytes | bytearray, projection: _Projection) -> pd.DataFrame:
    """Parse projected lexical fields while checking the full source record width."""
    names, columns, path = projection.names, projection.columns, projection.path
    if not payload:
        # A compressed range crossed by an earlier member owns no records.
        payload_table = pa.table({name: pa.array([], pa.string()) for name in columns})
        return _normalize(payload_table, features=projection.features, path=path)
    # Arrow checks full record widths but can accept malformed CSV quoting.
    try:
        with pa.BufferReader(payload) as source:
            table = pacsv.read_csv(
                source,
                read_options=pacsv.ReadOptions(column_names=names, use_threads=False),
                parse_options=pacsv.ParseOptions(newlines_in_values=False),
                convert_options=pacsv.ConvertOptions(
                    include_columns=columns, column_types=dict.fromkeys(columns, pa.string()), null_values=[]
                ),
            )
    except pa.ArrowInvalid as error:
        raise schema_error(
            path, TRANSCRIPTS, found=f"malformed transcript CSV: {error}", expected="one field per header column"
        ) from error
    return _normalize(table, features=projection.features, path=path)


def _plain_block(block: bytes, projection: _Projection, *, first: bool) -> pd.DataFrame:
    """Parse one newline-delimited byte block of an uncompressed source."""
    return _parse(block.partition(b"\n")[2] if first else block, projection)


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


def _partitions(projection: _Projection) -> list[Delayed]:
    """Return one delayed parser per plain block or compressed member range."""
    path = projection.path
    if path.name.endswith(".gz"):
        size = path.stat().st_size
        offset: int | Delayed = 0
        partitions: list[Delayed] = []
        for start in range(0, size, _GZIP_RANGE_BYTES):
            task = delayed(_gzip_range, nout=2)(
                projection, offset, start=start, stop=min(start + _GZIP_RANGE_BYTES, size)
            )
            offset, partition = task
            partitions.append(partition)
        return partitions

    # read_bytes has no typed signature; this is its documented result when include_path is false.
    read_blocks = cast("Callable[..., tuple[bytes | bool, list[list[Delayed]]]]", read_bytes)
    _, blocks = read_blocks([glob.escape(str(path))], delimiter=b"\n", blocksize=_BLOCK_BYTES, sample=False)
    return [delayed(_plain_block)(block, projection, first=index == 0) for index, block in enumerate(blocks[0])]


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
    names = tuple(read_header(path, TRANSCRIPTS))
    columns = transcript_columns(names, path)
    empty = pa.table({name: pa.array([], pa.string()) for name in columns})
    meta = _normalize(empty, features=features, path=path)
    frame = dd.from_delayed(_partitions(_Projection(names, columns, features, path)), meta=meta)

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
