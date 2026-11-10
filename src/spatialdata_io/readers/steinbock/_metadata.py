"""Read the supported fields of ``images.csv`` with their declared Arrow types.

One binary stream owns header inspection and the projected read. Arrow checks
every record's complete width and parses the supported fields of every row;
ignored extra values are not decoded or scanned for NUL. The schema validates
every image identity, then the dimensions and extents of the selected images.
The file is closed on success and failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.csv as pacsv

from spatialdata_io.readers._utils.csv import csv_errors, csv_header
from spatialdata_io.readers._utils.errors import add_reader_context
from spatialdata_io.readers.steinbock._schemas import (
    IMAGE_METADATA_CONTEXT,
    image_metadata_types,
    normalize_image_metadata,
)

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.steinbock._schemas import ImageMetadata


def read_image_metadata(path: ArtifactFile, selected: set[str]) -> dict[str, ImageMetadata]:
    """Return the normalized ``images.csv`` records of the ``selected`` image names."""
    try:
        with path.open("rb") as source, csv_errors(context=IMAGE_METADATA_CONTEXT, path=path):
            types = image_metadata_types(csv_header(path, source=source, context=IMAGE_METADATA_CONTEXT), path)
            source.seek(0)
            table = pacsv.read_csv(
                source,
                read_options=pacsv.ReadOptions(use_threads=False),
                parse_options=pacsv.ParseOptions(newlines_in_values=True),
                # Only an empty extent is missing; an empty image name stays an empty string.
                convert_options=pacsv.ConvertOptions(
                    column_types=types, include_columns=types.names, null_values=[""], strings_can_be_null=False
                ),
            )
    except OSError as error:
        add_reader_context(error, context=IMAGE_METADATA_CONTEXT, path=path)
        raise
    return normalize_image_metadata(table, selected, path)
