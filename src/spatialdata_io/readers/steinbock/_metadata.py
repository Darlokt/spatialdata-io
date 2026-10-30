"""Read the optional Steinbock IMC image-information sidecar once."""

from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers.steinbock._schemas import image_metadata_columns, normalize_image_metadata

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.steinbock._schemas import ImageMetadata

_CONTEXT = ReaderErrorContext("steinbock", "IMC images.csv")


def read_image_metadata(path: ArtifactFile, selected: set[str]) -> dict[str, ImageMetadata]:
    """Project source text under one file owner and enforce full CSV record widths."""
    try:
        with (
            path.open("rb") as source,
            io.TextIOWrapper(source, encoding="utf-8-sig", newline="") as text,
        ):
            records = csv.reader(text, strict=True)
            header = next(records, [])
            if any("\x00" in field for field in header):
                raise ReaderFormatError(
                    context=_CONTEXT,
                    path=path,
                    found="NUL-containing metadata",
                    expected="text fields without NUL bytes",
                )
            positions = image_metadata_columns(header, path)
            columns: dict[str, list[str]] = {name: [] for name in positions}
            for row in records:
                if any("\x00" in field for field in row):
                    raise ReaderFormatError(
                        context=_CONTEXT,
                        path=path,
                        found="NUL-containing metadata",
                        expected="text fields without NUL bytes",
                    )
                if len(row) != len(header):
                    raise ReaderFormatError(
                        context=_CONTEXT,
                        path=path,
                        found=f"record {records.line_num} with {len(row)} fields",
                        expected=f"exactly {len(header)} fields in every images.csv record",
                    )
                for name, index in positions.items():
                    columns[name].append(row[index])
    except (UnicodeError, csv.Error) as error:
        raise ReaderFormatError(
            context=_CONTEXT, path=path, found=str(error), expected="well-formed UTF-8 CSV metadata"
        ) from error
    except OSError as error:
        add_reader_context(error, context=_CONTEXT, path=path)
        raise
    return normalize_image_metadata(columns, selected, path)
