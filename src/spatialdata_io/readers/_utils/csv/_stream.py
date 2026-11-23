"""Own sequential CSV streams and report gzip corruption with reader context."""

from __future__ import annotations

import io
import zlib
from contextlib import contextmanager
from typing import TYPE_CHECKING, override

from spatialdata_io.readers._utils.errors import ReaderFormatError, add_reader_context

if TYPE_CHECKING:
    from collections.abc import Buffer, Generator
    from pathlib import Path
    from typing import BinaryIO

    from spatialdata_io.readers._utils.errors import ReaderErrorContext

_READ_BYTES = 128 * 1024
_GZIP_WINDOW = 31  # zlib's gzip mode validates the member header and trailer.


class _GzipStream(io.RawIOBase):
    """Adapt zlib's gzip member decoder to a bounded, sequential file stream."""

    def __init__(self, source: BinaryIO) -> None:
        super().__init__()
        self._source = source
        self._decoder = zlib.decompressobj(_GZIP_WINDOW)
        self._pending = b""

    @override
    def readable(self) -> bool:
        """Return whether this stream supports sequential reads."""
        return True

    def _next_member(self) -> bool:
        """Skip zero padding after a verified member and continue if another follows."""
        pending = self._pending.lstrip(b"\0")
        while not pending:
            chunk = self._source.read(_READ_BYTES)
            if not chunk:
                return False
            pending = chunk.lstrip(b"\0")
        self._pending = pending
        self._decoder = zlib.decompressobj(_GZIP_WINDOW)
        return True

    @override
    def readinto(self, buffer: Buffer, /) -> int:
        """Decode at most the caller's buffer size without reading a complete member."""
        view = memoryview(buffer).cast("B")
        if not view:
            return 0
        while True:
            if self._decoder.eof and not self._next_member():
                return 0
            chunk = self._pending or self._source.read(_READ_BYTES)
            decoded = self._decoder.decompress(chunk, len(view))
            self._pending = self._decoder.unused_data if self._decoder.eof else self._decoder.unconsumed_tail
            if decoded:
                view[: len(decoded)] = decoded
                return len(decoded)
            if not chunk and not self._decoder.eof:
                message = "compressed file ended before the end of a gzip member"
                raise EOFError(message)


@contextmanager
def open_csv_stream(
    path: Path, /, *, context: ReaderErrorContext, compressed: bool
) -> Generator[io.BufferedReader | io.BufferedReader[_GzipStream]]:
    """Open a binary CSV stream that closes on success or failure.

    The reader selects compression during artifact interpretation. Gzip input
    may contain concatenated members and zero padding. The decoder verifies
    each member as it reads; a caller reading just the CSV header need not
    reach the end of the first member.

    Raises
    ------
    ReaderFormatError
        If the gzip decoder encounters corruption while reading the stream.
    OSError
        If opening or reading the file fails, with reader and path context.
    """
    try:
        with path.open("rb") as source:
            if compressed:
                with io.BufferedReader(_GzipStream(source)) as stream:
                    yield stream
            else:
                yield source
    except (EOFError, zlib.error) as error:
        if not compressed:
            raise
        raise ReaderFormatError(
            context=context, path=path, found=f"invalid gzip data: {error}", expected="a readable gzip stream"
        ) from error
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise
