"""Construct one projected, path-backed CosMx transcript points element."""

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
from spatialdata.transformations import Identity

from spatialdata_io.readers.cosmx._schemas import (
    CELL,
    CELL_ID,
    FOV,
    INTEGER_DTYPE,
    STRING_DTYPE,
    float_values,
    integer_values,
    schema_error,
    transcript_columns,
)
from spatialdata_io.readers.cosmx._tables import read_header

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from dask.delayed import Delayed
    from spatialdata.transformations import BaseTransformation

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.cosmx._transforms import Placement

_BLOCK_BYTES = 8 * 1024 * 1024
_GZIP_RANGE_BYTES = 16 * 1024 * 1024
_GZIP_MAGIC = b"\x1f\x8b\x08"
_GZIP_RESERVED_FLAGS = 0xE0
_GZIP_HEADER_BYTES = 10
_READ_BYTES = 1024 * 1024
# A first member larger than this is not the independently compressed layout.
_MAX_MEMBER_BYTES = 64 * 1024 * 1024


def _normalize(table: pa.Table, *, features: tuple[str, ...], path: ArtifactFile) -> pd.DataFrame:
    """Validate one lexical partition and convert it to the returned point schema."""
    target = pd.Series(table.column("target").to_pandas(types_mapper=pd.ArrowDtype), dtype=STRING_DTYPE)
    if target.isna().any() or not target.isin(features).all():
        raise schema_error(path, found="missing or unknown targets", expected="targets from the count vocabulary")

    frame = pd.DataFrame(
        {
            "x": float_values(table.column("x_global_px"), path, "x_global_px"),
            "y": float_values(table.column("y_global_px"), path, "y_global_px"),
            "target": pd.Categorical(target.to_numpy(), categories=features),
            FOV: pd.Series(integer_values(table.column(FOV), path, FOV, minimum=1), dtype=INTEGER_DTYPE),
            CELL_ID: pd.Series(integer_values(table.column(CELL_ID), path, CELL_ID, minimum=0), dtype=INTEGER_DTYPE),
            "x_local": float_values(table.column("x_local_px"), path, "x_local_px"),
            "y_local": float_values(table.column("y_local_px"), path, "y_local_px"),
        }
    )
    if "z" in table.column_names:
        # The vendor defines z as a slice number in arbitrary units; it can be negative.
        frame["z_raw"] = pd.Series(integer_values(table.column("z"), path, "z", minimum=None), dtype=INTEGER_DTYPE)
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


def _parse(payload: bytes, projection: _Projection) -> pd.DataFrame:
    """Parse projected lexical fields while checking the full source record width."""
    names, columns, path = projection.names, projection.columns, projection.path
    if not payload:
        # A compressed range inside a member that began earlier owns no records.
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
            path, found=f"malformed transcript CSV: {error}", expected="one field per header column"
        ) from error
    return _normalize(table, features=projection.features, path=path)


def _without_header(payload: bytes) -> bytes:
    """Drop the header record that begins the first source partition."""
    return payload.split(b"\n", 1)[1] if b"\n" in payload else b""


def _plain_block(block: bytes, projection: _Projection, *, first: bool) -> pd.DataFrame:
    """Parse one newline-delimited byte block of an uncompressed source."""
    return _parse(_without_header(block) if first else block, projection)


def _decode_member(handle: BinaryIO, offset: int, path: ArtifactFile) -> tuple[bytes, int] | None:
    """Decode one complete gzip member, or return None if ``offset`` does not start one.

    zlib verifies the deflate stream end, CRC32 and size trailer, so a byte pattern
    inside compressed data is practically never mistaken for a member start.
    """
    handle.seek(offset)
    decoder = zlib.decompressobj(31)
    parts: list[bytes] = []
    consumed = 0
    while not decoder.eof:
        chunk = handle.read(_READ_BYTES)
        if not chunk:
            return None
        try:
            parts.append(decoder.decompress(chunk))
        except zlib.error:
            return None
        consumed += len(chunk)
        if consumed > _MAX_MEMBER_BYTES and not decoder.eof:
            raise schema_error(
                path,
                found=f"a gzip member at byte {offset} larger than {_MAX_MEMBER_BYTES} compressed bytes",
                expected="independently compressed members of bounded size",
            )
    return b"".join(parts), consumed - len(decoder.unused_data)


def _first_member(handle: BinaryIO, start: int, stop: int, path: ArtifactFile) -> tuple[int, bytes, int] | None:
    """Find the first verified member starting in ``[start, stop)``."""
    position = start
    while position < stop:
        handle.seek(position)
        # Each window owns candidates in its first _READ_BYTES and sees their whole header.
        window = handle.read(_READ_BYTES + _GZIP_HEADER_BYTES)
        index = window.find(_GZIP_MAGIC)
        while index != -1 and index < _READ_BYTES and position + index < stop:
            if index + 3 < len(window) and window[index + 3] & _GZIP_RESERVED_FLAGS == 0:
                decoded = _decode_member(handle, position + index, path)
                if decoded is not None:
                    return position + index, *decoded
            index = window.find(_GZIP_MAGIC, index + 1)
        if len(window) <= _READ_BYTES:
            return None
        position += _READ_BYTES
    return None


@dataclass(frozen=True, slots=True)
class _MemberRange:
    """Immutable task state: a compressed byte range, ending at most at the file end."""

    path: Path
    start: int
    stop: int


def _gzip_range(spec: _MemberRange, projection: _Projection) -> pd.DataFrame:
    """Decode the whole members starting in one compressed range; each must end a record."""
    path = projection.path
    payloads: list[bytes] = []
    with spec.path.open("rb") as handle:
        found = _first_member(handle, spec.start, spec.stop, path)
        while found is not None:
            offset, payload, size = found
            if not payload.endswith(b"\n"):
                raise schema_error(
                    path,
                    found=f"gzip member at byte {offset} ending inside a record",
                    expected="record-aligned members",
                )
            payloads.append(payload)
            offset += size
            if offset >= spec.stop:
                break
            # Members are contiguous, so a member must start exactly where the previous one ended.
            decoded = _decode_member(handle, offset, path)
            if decoded is None:
                raise schema_error(path, found=f"corrupt gzip member at byte {offset}", expected="complete members")
            found = (offset, *decoded)
    data = b"".join(payloads)
    return _parse(_without_header(data) if spec.start == 0 else data, projection)


def _gzip_ranges(path: ArtifactFile) -> list[_MemberRange]:
    """Validate the member-aligned layout cheaply and split it into compressed ranges."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        first = _decode_member(handle, 0, path)
    expected = "a gzip file of independently compressed record-aligned members"
    if first is None:
        raise schema_error(path, found="an oversized, corrupt or non-gzip first member", expected=expected)
    if first[1] >= size:
        raise schema_error(
            path, found="a single-stream gzip file", expected=f"{expected}; decompress it or recompress it in blocks"
        )
    return [
        _MemberRange(path, start, min(start + _GZIP_RANGE_BYTES, size)) for start in range(0, size, _GZIP_RANGE_BYTES)
    ]


def _partitions(projection: _Projection) -> list[Delayed]:
    """Return one delayed parser per plain block or compressed member range."""
    path = projection.path
    if path.name.endswith(".gz"):
        return [delayed(_gzip_range)(spec, projection) for spec in _gzip_ranges(path)]

    # read_bytes has no typed signature; this is its documented result when include_path is false.
    read_blocks = cast("Callable[..., tuple[bytes | bool, list[list[Delayed]]]]", read_bytes)
    _, blocks = read_blocks([glob.escape(str(path))], delimiter=b"\n", blocksize=_BLOCK_BYTES, sample=False)
    return [delayed(_plain_block)(block, projection, first=index == 0) for index, block in enumerate(blocks[0])]


def read_transcripts(
    path: ArtifactFile, *, features: tuple[str, ...], placements: Mapping[int, Placement]
) -> dd.DataFrame:
    """Return global XY points with every FOV frame, without reading transcript rows.

    Records must each occupy one physical line. Values are validated when their
    partition is computed.
    """
    names = tuple(read_header(path))
    columns = transcript_columns(names, path)
    empty = pa.table({name: pa.array([], pa.string()) for name in columns})
    meta = _normalize(empty, features=features, path=path)
    frame = dd.from_delayed(_partitions(_Projection(names, columns, features, path)), meta=meta)

    transformations: dict[str, BaseTransformation] = {"global": Identity()}
    transformations.update({str(fov): placement.global_to_local() for fov, placement in placements.items()})
    # Independent partitions are deliberately unsorted. Known target categories and
    # sort=False avoid the public model's eager category and index scans.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The index of the dataframe is not monotonic increasing.*", category=UserWarning
        )
        return PointsModel.parse(
            frame, coordinates={"x": "x", "y": "y"}, feature_key="target", transformations=transformations, sort=False
        )
