"""Read optional experiment and channel annotations for one QPI scan.

A Fusion XPD is a UTF-8 JSON protocol. It contributes its cycle count,
available format version and distinct marker count. A marker-list CSV also
contributes marker counts. Both record how many distinct names occur in the
measured scan and warn if a nonempty list disagrees. A HuBMAP channel CSV
instead contributes its validated identifiers and segmentation and antibody
flags in submission order. These annotations leave the image channels intact.

Each sidecar is read once with a 4 MiB limit and closed before parsing.
The result contains only the selected protocol summaries or annotation fields.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.csv as pacsv

from spatialdata_io.readers._utils.csv import csv_errors, csv_header, open_csv_stream
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context, warn_user
from spatialdata_io.readers._utils.metadata import parse_json_object
from spatialdata_io.readers.phenocycler._metadata import _marker, fusion_protocol
from spatialdata_io.readers.phenocycler._schemas import channel_schema, hubmap_annotations

if TYPE_CHECKING:
    from collections.abc import Iterable

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.phenocycler._layout import RegionFiles

_SIDECAR_BYTES = 4 * 1024 * 1024


def _check_size(data: bytes, path: ArtifactFile, context: ReaderErrorContext) -> None:
    if len(data) > _SIDECAR_BYTES:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"metadata exceeding {_SIDECAR_BYTES} bytes",
            expected=f"a sidecar at most {_SIDECAR_BYTES} bytes long",
        )


def _xpd(path: ArtifactFile, context: ReaderErrorContext) -> tuple[dict[str, int], set[str]]:
    try:
        with path.open("rb") as source:
            data = source.read(_SIDECAR_BYTES + 1)
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise
    _check_size(data, path, context)
    try:
        decoded = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ReaderFormatError(
            context=context, path=path, found="invalid UTF-8", expected="a UTF-8 XPD protocol JSON object"
        ) from error
    content = parse_json_object(decoded, context=context, path=path)
    protocol = fusion_protocol(content)
    if "wells" in content and not isinstance(content["wells"], list):
        raise ReaderFormatError(
            context=context, path=path, found="a non-array wells field", expected="an array of Fusion protocol wells"
        )
    details = {"cycle_count": protocol.cycles}
    if protocol.version is not None:
        details["format_version"] = protocol.version
    return details, {channel.marker for channel in protocol.channels}


def _channels_csv(path: ArtifactFile, context: ReaderErrorContext, names: tuple[str, ...]) -> dict[str, object]:
    """Read a bounded marker list or HuBMAP panel and project its supported fields."""
    with open_csv_stream(path, context=context, compressed=False) as source:
        data = source.read(_SIDECAR_BYTES + 1)
    _check_size(data, path, context)
    header = csv_header(path, source=io.BytesIO(data), context=context)
    schema = channel_schema(header, context=context, path=path)
    with csv_errors(context=context, path=path):
        rows = pacsv.read_csv(
            pa.py_buffer(data),
            read_options=pacsv.ReadOptions(use_threads=False, block_size=_SIDECAR_BYTES),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(
                column_types=schema,
                include_columns=schema.names,
                null_values=[],
                strings_can_be_null=False,
                true_values=["Yes", "YES", "yes"],
                false_values=["No", "NO", "no"],
            ),
        )
    if len(schema) > 1:
        return {"annotations": hubmap_annotations(rows, len(names), context=context, path=path)}
    markers: set[str] = set()
    for value in rows.column(0):
        if (name := _marker(value.as_py())) is not None:
            markers.add(name)
    return dict(_summary(markers, set(names), context))


def _summary(markers: set[str], scan_markers: set[str], context: ReaderErrorContext) -> dict[str, int]:
    if markers and markers != scan_markers:
        warn_user(
            f"{context.artifact} contains {len(markers)} distinct markers, of which "
            f"{len(markers & scan_markers)} match the {len(scan_markers)} QPI marker names."
        )
    return {"marker_count": len(markers), "matching_marker_count": len(markers & scan_markers)}


def read_sidecars(
    files: RegionFiles, marker_names: Iterable[str], *, detected_version: str | None
) -> dict[str, object]:
    """Read the selected sidecars and retain panel annotations or marker agreement summaries."""
    result: dict[str, object] = {}
    names = tuple(marker_names)
    if files.xpd is not None:
        context = ReaderErrorContext("phenocycler", "XPD protocol", detected_version)
        details, markers = _xpd(files.xpd, context)
        result["xpd"] = details | _summary(markers, set(names), context)
    if files.channels_csv is not None:
        context = ReaderErrorContext("phenocycler", "channels.csv", detected_version)
        result["channels_csv"] = _channels_csv(files.channels_csv, context, names)
    return result
