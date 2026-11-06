"""Strict first-record CSV header parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.csv import parse_csv_header, read_csv_header
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

_CONTEXT = ReaderErrorContext("test", "cells")


def _file(tmp_path: Path, payload: bytes) -> ArtifactFile:
    path = ArtifactFile(tmp_path / "cells.csv")
    path.write_bytes(payload)
    return path


class TestParseCsvHeader:
    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            (b"a,b,c\n", ["a", "b", "c"]),
            (b"\xef\xbb\xbfa,b\r\n", ["a", "b"]),
            (b'"x,y",b,"",b\n', ["x,y", "b", "", "b"]),
            ("gène,µm\n".encode(), ["gène", "µm"]),
            (b"", []),
        ],
    )
    def test_parses_one_record_without_repairing_names(self, tmp_path: Path, line: bytes, expected: list[str]) -> None:
        assert parse_csv_header(line, context=_CONTEXT, path=tmp_path) == expected

    def test_uses_the_delimiter(self, tmp_path: Path) -> None:
        assert parse_csv_header(b"a,b\tc\n", context=_CONTEXT, path=tmp_path, delimiter="\t") == ["a,b", "c"]

    @pytest.mark.parametrize(
        ("line", "found"),
        [
            (b'"a,b\n', r"malformed CSV header \(unexpected end of data\)"),
            (b'"a"b,c\n', r"malformed CSV header \('.' expected after"),
            (b"a,\xff\n", r"malformed CSV header \(not valid UTF-8\)"),
        ],
    )
    def test_rejects_malformed_headers_with_context(self, tmp_path: Path, line: bytes, found: str) -> None:
        with pytest.raises(ReaderFormatError, match=found) as raised:
            parse_csv_header(line, context=_CONTEXT, path=tmp_path)

        assert raised.value.context == _CONTEXT
        assert raised.value.path == tmp_path


class TestReadCsvHeader:
    def test_reads_only_the_first_physical_line(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _file(tmp_path, b"a,b\n1,2\n" + b"3,4\n" * 1000)
        opened = mocker.spy(type(path), "open")

        assert read_csv_header(path, context=_CONTEXT) == ["a", "b"]

        (stream,) = opened.spy_return_list
        assert stream.closed

    def test_quoted_newline_in_a_header_is_malformed(self, tmp_path: Path) -> None:
        path = _file(tmp_path, b'"a\nb",c\n1,2\n')

        with pytest.raises(ReaderFormatError, match="malformed CSV header") as raised:
            read_csv_header(path, context=_CONTEXT)

        assert raised.value.path == path

    def test_io_failure_keeps_its_type_with_context(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "missing.csv")

        with pytest.raises(FileNotFoundError) as raised:
            read_csv_header(path, context=_CONTEXT)

        assert raised.value.__notes__ == [f"While handling {_CONTEXT.describe(path=path)}."]
