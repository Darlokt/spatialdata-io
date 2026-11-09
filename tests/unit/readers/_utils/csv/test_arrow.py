"""Arrow header parsing and Arrow CSV errors with reader context."""

from __future__ import annotations

import gzip
import io
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.csv as pacsv
import pytest

from spatialdata_io.readers._utils.csv import csv_errors, csv_header
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


class TestCsvHeader:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (b"a,b,c\n1,2,3\n", ("a", "b", "c")),
            (b"\xef\xbb\xbfa,b\r\n1,2\r\n", ("a", "b")),
            (b'"x,y","q""q",,b\n', ("x,y", 'q"q', "", "b")),
            (b" a ,b\n", (" a ", "b")),
            ("gène,µm\n".encode(), ("gène", "µm")),
        ],
    )
    def test_reads_names_with_arrow_grammar(self, tmp_path: Path, payload: bytes, expected: tuple[str, ...]) -> None:
        assert csv_header(_file(tmp_path, payload), context=_CONTEXT) == expected

    def test_reads_only_the_first_line_of_a_stream(self, tmp_path: Path) -> None:
        path = tmp_path / "cells.csv.gz"
        path.write_bytes(gzip.compress(b"a,b\n1,2\n"))
        with gzip.open(path, "rb") as source:
            assert csv_header(path, context=_CONTEXT, source=source) == ("a", "b")
            assert source.read() == b"1,2\n"

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (b"gene\tcluster\tavg_log2FC\n", ("gene", "cluster", "avg_log2FC")),
            (b"\xef\xbb\xbf\tMoransI_observed\tMoransI_p.value\r\n", ("", "MoransI_observed", "MoransI_p.value")),
            (b'"gene\tlabel"\t"q""q"\tcomma,name\n', ("gene\tlabel", 'q"q', "comma,name")),
        ],
    )
    def test_named_tsv_headers(self, tmp_path: Path, payload: bytes, expected: tuple[str, ...]) -> None:
        with io.BytesIO(payload + b"unread record\n") as source:
            assert csv_header(tmp_path / "features.tsv", context=_CONTEXT, source=source, delimiter="\t") == expected
            assert not source.closed
            assert source.read() == b"unread record\n"

    def test_duplicate_tsv_names(self, tmp_path: Path) -> None:
        path = _file(tmp_path, b"gene\tstat\tstat\n")
        with pytest.raises(ReaderFormatError, match="repeated column names") as raised:
            csv_header(path, context=_CONTEXT, delimiter="\t")
        assert raised.value.path == path
        assert raised.value.context == _CONTEXT

    @pytest.mark.parametrize(
        ("payload", "found"),
        [
            (b"a,b,a,c,b\n", r"repeated column names \['a', 'b'\]"),
            (b"", r"malformed CSV \(Empty CSV file\)"),
            (b"a,b", r"malformed CSV \(CSV parse error: Empty CSV file or block"),
            (b'"a\nb",c\n', r"malformed CSV \(CSV parse error: Empty CSV file or block"),
            (b"a,\xff\n", r"malformed CSV \('utf-8' codec can't decode byte 0xff"),
        ],
    )
    def test_rejects_unusable_headers_with_context(self, tmp_path: Path, payload: bytes, found: str) -> None:
        path = _file(tmp_path, payload)
        with pytest.raises(ReaderFormatError, match=found) as raised:
            csv_header(path, context=_CONTEXT)

        assert raised.value.context == _CONTEXT
        assert raised.value.path == path

    def test_names_the_artifact_of_an_unreadable_file(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _file(tmp_path, b"a\n")
        mocker.patch.object(type(path), "open", side_effect=PermissionError(13, "denied"))
        with pytest.raises(PermissionError) as raised:
            csv_header(path, context=_CONTEXT)

        assert raised.value.__notes__ == [f"While handling {_CONTEXT.describe(path=path)}."]

    def test_stream_failures_keep_context_and_caller_ownership(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "cells.csv.gz"
        with io.BytesIO() as source:
            mocker.patch.object(source, "readline", side_effect=OSError(5, "read failed"))
            with pytest.raises(OSError, match="read failed") as raised:
                csv_header(path, context=_CONTEXT, source=source)

            assert not source.closed
            assert raised.value.__notes__ == [f"While handling {_CONTEXT.describe(path=path)}."]


class TestCsvErrors:
    @pytest.mark.parametrize(
        ("payload", "found"),
        [
            (b"a,b\n1\n", "Expected 2 columns, got 1"),
            (b"a,b\n1,-1\n", "CSV conversion error to uint64: invalid value '-1'"),
            (b"a,b\n\xff,1\n", "invalid UTF8 data"),
        ],
    )
    def test_reports_arrow_failures_as_reader_errors(self, tmp_path: Path, payload: bytes, found: str) -> None:
        path = _file(tmp_path, payload)
        options = pacsv.ConvertOptions(column_types={"a": pa.string(), "b": pa.uint64()}, null_values=[])
        with pytest.raises(ReaderFormatError, match=found) as raised, csv_errors(context=_CONTEXT, path=path):
            pacsv.read_csv(path, convert_options=options)

        assert raised.value.context == _CONTEXT
        assert raised.value.path == path
        assert isinstance(raised.value.__cause__, pa.ArrowInvalid)

    def test_passes_other_errors_and_results_through(self, tmp_path: Path) -> None:
        path = _file(tmp_path, b"a\n1\n")
        with csv_errors(context=_CONTEXT, path=path):
            table = pacsv.read_csv(io.BytesIO(path.read_bytes()))
        assert table.column("a").to_pylist() == [1]
        unrelated = KeyError("a")
        with pytest.raises(KeyError) as raised, csv_errors(context=_CONTEXT, path=path):
            raise unrelated
        assert raised.value is unrelated
