"""Read ``images.csv`` once, streaming its records through the schema in ``_schemas``.

The file is opened in binary mode and decoded as UTF-8 (an optional byte-order
mark is removed) by one text wrapper, and parsed by the standard ``csv`` module
in strict mode, so an unterminated or misplaced quote fails instead of being
repaired. Every record is checked for NUL bytes and for the header's width,
including records of images that were not discovered; only the used fields are
projected. The file is closed on success and failure.
"""

from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderFormatError, add_reader_context
from spatialdata_io.readers.steinbock._schemas import (
    IMAGE_METADATA_CONTEXT,
    image_metadata_columns,
    normalize_image_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.steinbock._schemas import ImageMetadata


def read_image_metadata(path: ArtifactFile, selected: set[str]) -> dict[str, ImageMetadata]:
    """Return the normalized ``images.csv`` records of the ``selected`` image names."""
    try:
        with (
            path.open("rb") as source,
            io.TextIOWrapper(source, encoding="utf-8-sig", newline="") as text,
        ):
            return normalize_image_metadata(_records(text, path), selected, path)
    except (UnicodeError, csv.Error) as error:
        raise ReaderFormatError(
            context=IMAGE_METADATA_CONTEXT, path=path, found=str(error), expected="well-formed UTF-8 CSV metadata"
        ) from error
    except OSError as error:
        add_reader_context(error, context=IMAGE_METADATA_CONTEXT, path=path)
        raise


def _records(text: Iterable[str], path: ArtifactFile) -> Iterator[dict[str, str]]:
    """Yield each record's used fields after checking all of its fields."""
    records = csv.reader(text, strict=True)
    header = next(records, [])
    _reject_nul(header, path)
    positions = image_metadata_columns(header, path)
    for row in records:
        _reject_nul(row, path)
        if len(row) != len(header):
            raise ReaderFormatError(
                context=IMAGE_METADATA_CONTEXT,
                path=path,
                found=f"record {records.line_num} with {len(row)} fields",
                expected=f"exactly {len(header)} fields in every images.csv record",
            )
        yield {name: row[index] for name, index in positions.items()}


def _reject_nul(fields: list[str], path: ArtifactFile) -> None:
    """Reject a NUL byte in a field; the ``csv`` module accepts them."""
    if any("\x00" in field for field in fields):
        raise ReaderFormatError(
            context=IMAGE_METADATA_CONTEXT,
            path=path,
            found="NUL-containing metadata",
            expected="text fields without NUL bytes",
        )
