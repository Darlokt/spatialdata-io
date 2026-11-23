"""Own a Xenium Parquet file for footer inspection or projected payload reading."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from spatialdata_io.readers._utils.errors import ReaderFormatError, add_reader_context

if TYPE_CHECKING:
    from collections.abc import Generator

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.paths import ArtifactFile


@contextmanager
def open_parquet(path: ArtifactFile, *, context: ReaderErrorContext) -> Generator[pq.ParquetFile]:
    """Keep footer and payload reads under one source owner and attach artifact context."""
    try:
        with pq.ParquetFile(path) as source:
            yield source
    except pa.ArrowInvalid as error:
        raise ReaderFormatError(
            context=context, path=path, found=f"malformed Parquet ({error})", expected="a readable Parquet artifact"
        ) from error
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise
