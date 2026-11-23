"""Sequential plain and gzip CSV stream ownership and errors."""

from __future__ import annotations

import errno
import gzip
import io
import zlib
from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.csv import _stream, csv_header, open_csv_stream
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

_CONTEXT = ReaderErrorContext("test", "cells", "2.0")
_HEADER = b"id,value\n"
_ROWS = b"1,2\n3,4\n"
_MEMBER = gzip.compress(_ROWS, mtime=0)
_BAD_CRC = _MEMBER[:-8] + bytes([_MEMBER[-8] ^ 1]) + _MEMBER[-7:]
_WITH_FHCRC = _MEMBER[:3] + b"\x02" + _MEMBER[4:10]
_VALID_HEADER_CRC = (zlib.crc32(_WITH_FHCRC) & 0xFFFF).to_bytes(2, "little")
_BAD_HEADER_CRC = bytes([_VALID_HEADER_CRC[0] ^ 1]) + _VALID_HEADER_CRC[1:]


class TestOpenCsvStream:
    @pytest.mark.parametrize(
        ("payload", "compressed"),
        [(_HEADER + _ROWS, False), (gzip.compress(_HEADER, mtime=0) + _MEMBER + b"\0" * 16, True)],
        ids=["plain", "padded-members"],
    )
    def test_header_and_records_share_a_stream(self, tmp_path: Path, payload: bytes, *, compressed: bool) -> None:
        # The canonical target of an artifact can have any suffix.
        path = tmp_path / "artifact"
        path.write_bytes(payload)

        with open_csv_stream(path, context=_CONTEXT, compressed=compressed) as stream:
            assert csv_header(path, context=_CONTEXT, source=stream) == ("id", "value")
            assert stream.read() == _ROWS

        assert stream.closed

    def test_padded_members_and_headers_can_cross_input_blocks(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch.object(_stream, "_READ_BYTES", 8)
        path = tmp_path / "cells.csv.gz"
        first = gzip.compress(_HEADER, mtime=0)
        second = _WITH_FHCRC + _VALID_HEADER_CRC + _MEMBER[10:]
        path.write_bytes(first + b"\0" * 32 + second + b"\0" * 64)

        with open_csv_stream(path, context=_CONTEXT, compressed=True) as stream:
            assert stream.readline() == _HEADER
            assert stream.read(3) == _ROWS[:3]
            assert stream.read() == _ROWS[3:]

    @pytest.mark.parametrize(
        "member",
        [
            _MEMBER[:-4],
            _BAD_CRC,
            _WITH_FHCRC + _BAD_HEADER_CRC + _MEMBER[10:],
            _MEMBER[:3] + b"\x20" + _MEMBER[4:],
            b"unexpected",
        ],
        ids=["truncated", "crc", "header-crc", "reserved-flag", "trailing"],
    )
    def test_corruption_is_contextual_and_closes_stream(self, tmp_path: Path, member: bytes) -> None:
        path = tmp_path / "cells.csv.gz"
        path.write_bytes(gzip.compress(_HEADER, mtime=0) + member)

        with (
            pytest.raises(ReaderFormatError, match="invalid gzip data") as caught,
            open_csv_stream(path, context=_CONTEXT, compressed=True) as stream,
        ):
            stream.read()

        assert caught.value.context == _CONTEXT
        assert caught.value.path == path
        assert stream.closed

    @pytest.mark.parametrize("compressed", [False, True])
    def test_missing_file_retains_os_error(self, tmp_path: Path, *, compressed: bool) -> None:
        path = tmp_path / "missing"
        with pytest.raises(FileNotFoundError) as caught, open_csv_stream(path, context=_CONTEXT, compressed=compressed):
            pytest.fail("opening a missing file succeeded")

        assert caught.value.errno == errno.ENOENT
        assert caught.value.filename == str(path)
        assert caught.value.__notes__ == [f"While handling {_CONTEXT.describe(path=path)}."]

    def test_eof_error_in_plain_stream_is_preserved(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "cells.csv"
        source = io.BytesIO()
        failed = EOFError("injected read failure")
        mocker.patch.object(type(path), "open", return_value=source)
        mocker.patch.object(source, "read", side_effect=failed)
        with pytest.raises(EOFError) as caught, open_csv_stream(path, context=_CONTEXT, compressed=False) as stream:
            stream.read()

        assert caught.value is failed
        assert stream.closed

    def test_header_io_error_gets_one_context_note(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "cells.csv"
        source = io.BytesIO()
        failed = OSError(errno.EIO, "injected read failure")
        mocker.patch.object(type(path), "open", return_value=source)
        mocker.patch.object(source, "readline", side_effect=failed)

        with (
            pytest.raises(OSError, match="injected read failure") as caught,
            open_csv_stream(path, context=_CONTEXT, compressed=False) as stream,
        ):
            csv_header(path, context=_CONTEXT, source=stream)

        assert caught.value is failed
        assert caught.value.__notes__ == [f"While handling {_CONTEXT.describe(path=path)}."]
        assert stream.closed
