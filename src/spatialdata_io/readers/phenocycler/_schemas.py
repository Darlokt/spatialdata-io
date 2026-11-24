"""Define the named Arrow fields of QPI channel annotation files.

The HuBMAP PhenoCycler template has four required columns. The string
``channel id`` preserves submitted identifiers such as ``Channel:0:13``.
The other three fields record Yes/No answers for nuclei segmentation, cell
segmentation and antibody status. Each measured channel requires one row
with a distinct, nonempty identifier and all three flags. The output keeps
submission order. A row's position cannot establish its QPI biomarker.

Marker-list exports instead supply a ``Marker`` column for comparison with
the scan's marker set. Both schemas project fields by name, ignore extra
columns and reject ambiguous spellings of a required name after trimming
whitespace and folding case.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

import pyarrow as pa

from spatialdata_io.readers._utils.errors import ReaderFormatError

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.errors import ReaderErrorContext

HUBMAP_CHANNEL_FIELDS = pa.schema(
    [
        ("channel id", pa.string()),
        ("is channel used for nuclei segmentation", pa.bool_()),
        ("is channel used for cell segmentation", pa.bool_()),
        ("is antibody", pa.bool_()),
    ]
)


def channel_schema(header: tuple[str, ...], *, context: ReaderErrorContext, path: Path) -> pa.Schema:
    """Project the HuBMAP channel annotations or the marker list using the file's actual field names.

    HuBMAP defines four required fields and allows additional columns. Resolve the
    fields by name so column order cannot change their meaning. Marker-list exports have a single
    recognized field, ``Marker``. Both layouts accept surrounding whitespace and
    case variants in the header. Two spellings of a selected name are ambiguous.
    """
    names = {column.strip().casefold() for column in header}
    hubmap = bool(names.intersection(HUBMAP_CHANNEL_FIELDS.names[1:])) or (
        "channel id" in names and "marker" not in names
    )
    required = HUBMAP_CHANNEL_FIELDS if hubmap else pa.schema([("marker", pa.string())])
    fields = []
    for field in required:
        matches = [column for column in header if column.strip().casefold() == field.name]
        if len(matches) != 1:
            expected = (
                "one named Marker column or the four HuBMAP channel fields"
                if not hubmap
                else f"one of each HuBMAP field {required.names!r}"
            )
            raise ReaderFormatError(
                context=context, path=path, found=f"{len(matches)} fields matching {field.name!r}", expected=expected
            )
        fields.append((matches[0], field.type))
    return pa.schema(fields)


def hubmap_annotations(
    rows: pa.Table, channel_count: int, *, context: ReaderErrorContext, path: Path
) -> list[dict[str, str | bool]]:
    """Validate one annotation per measured channel and preserve the submitter's IDs.

    The submitted channel IDs may use a different convention from QPI biomarker
    names. Keep the source rows and their order as independent metadata.
    """
    if len(rows) != channel_count:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"{len(rows)} channel rows",
            expected=f"one row for each of the {channel_count} QPI channels",
        )
    if any(column.null_count for column in rows.columns):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="null channel annotations",
            expected="a channel identifier and three Yes/No values in every row",
        )
    identifiers = rows.column(0).to_pylist()
    if len(set(identifiers)) != channel_count or any(
        not name.strip() or "\x00" in name or PureWindowsPath(name).is_absolute() or PurePosixPath(name).is_absolute()
        for name in identifiers
    ):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="empty, repeated or invalid channel IDs",
            expected="distinct nonempty channel identifiers",
        )
    return rows.rename_columns(HUBMAP_CHANNEL_FIELDS.names).to_pylist()
