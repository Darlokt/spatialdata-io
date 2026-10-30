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
from spatialdata.transformations import Identity, Scale

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
            path, found=f"malformed transcript CSV: {error}", expected="one field per header column"
        ) from error
    return _normalize(table, features=projection.features, path=path)


def _without_header(payload: bytes) -> bytes:
    """Drop the header record that begins the first source partition."""
    return payload.split(b"\n", 1)[1] if b"\n" in payload else b""


def _plain_block(block: bytes, projection: _Projection, *, first: bool) -> pd.DataFrame:
    """Parse one newline-delimited byte block of an uncompressed source."""
    return _parse(_without_header(block) if first else block, projection)


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
            raise schema_error(path, found=f"corrupt gzip member at byte {offset}", expected="complete members")
        try:
            decoded = decoder.decompress(chunk, _READ_BYTES)
        except zlib.error as error:
            raise schema_error(
                path, found=f"corrupt gzip member at byte {offset}: {error}", expected="complete members"
            ) from error
        payload.extend(decoded)
        pending = decoder.unconsumed_tail
    if len(payload) == member_start or not payload.endswith(b"\n"):
        raise schema_error(
            path, found=f"gzip member at byte {offset} ending inside a record", expected="record-aligned members"
        )
    return handle.tell() - len(decoder.unused_data), decoder.unused_data


@dataclass(frozen=True, slots=True)
class _MemberRange:
    """Immutable task state: a compressed byte range, ending at most at the file end."""

    start: int
    stop: int


def _gzip_range(spec: _MemberRange, projection: _Projection, offset: int) -> tuple[int, pd.DataFrame]:
    """Read contiguously from the predecessor's verified boundary, never resynchronizing.

    The scalar end offset orders source reads without keeping open resources in
    the graph. A member crossing several ranges belongs wholly to its first task;
    those subsequent ranges return empty partitions.
    """
    payload = bytearray()
    pending = b""
    with projection.path.open("rb") as handle:
        while offset < spec.stop:
            offset, pending = _decode_member(handle, offset, projection.path, payload, pending)
    if spec.start == 0:
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
            spec = _MemberRange(start, min(start + _GZIP_RANGE_BYTES, size))
            task = delayed(_gzip_range, nout=2)(spec, projection, offset)
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
    names = tuple(read_header(path))
    columns = transcript_columns(names, path)
    empty = pa.table({name: pa.array([], pa.string()) for name in columns})
    meta = _normalize(empty, features=features, path=path)
    frame = dd.from_delayed(_partitions(_Projection(names, columns, features, path)), meta=meta)

    transformations: dict[str, BaseTransformation] = {
        "global": Identity() if pixel_size_um is None else Scale([pixel_size_um, pixel_size_um], axes=("x", "y"))
    }
    transformations.update({str(fov): placement.global_to_local() for fov, placement in placements.items()})
    # Partitions are deliberately unsorted. Known target categories and
    # sort=False avoid the public model's eager category and index scans.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The index of the dataframe is not monotonic increasing.*", category=UserWarning
        )
        return PointsModel.parse(
            frame, coordinates={"x": "x", "y": "y"}, feature_key="target", transformations=transformations, sort=False
        )
