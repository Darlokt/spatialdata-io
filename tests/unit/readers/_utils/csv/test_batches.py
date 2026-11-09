"""Eager Arrow text record batches and tables from a caller-owned CSV stream."""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.csv as pacsv
import pytest

from spatialdata_io.readers._utils.csv import _batches, iter_csv_batches, read_csv_blocks, read_csv_table
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture

_CONTEXT = ReaderErrorContext("test", "counts")
_PATH = Path("counts.csv")
_HEADER = ("id", "gene", "x")
_TEXT = b'id,gene,x\n9007199254740993,"A,B",\n0012,"line\nbreak",0.10000000000000002\n3,"say ""hi""",-0\n'


def _rows(table: pa.Table | pa.RecordBatch) -> list[dict[str, str]]:
    return table.to_pylist()


_EXPECTED = [
    {"id": "9007199254740993", "gene": "A,B", "x": ""},
    {"id": "0012", "gene": "line\nbreak", "x": "0.10000000000000002"},
    {"id": "3", "gene": 'say "hi"', "x": "-0"},
]


class TestIterCsvBatches:
    def test_streams_exact_text_after_the_header(self) -> None:
        source = io.BytesIO(_TEXT)

        batches = list(iter_csv_batches(source, header=_HEADER, context=_CONTEXT, path=_PATH))

        assert [row for batch in batches for row in _rows(batch)] == _EXPECTED
        assert all(batch.schema == pa.schema([(name, pa.string()) for name in _HEADER]) for batch in batches)
        assert all(batch.column(name).null_count == 0 for batch in batches for name in _HEADER)
        assert not source.closed

    def test_block_bytes_bound_the_batches(self) -> None:
        text = b"id,gene,x\n" + b"".join(f"{index},G{index},{index}.5\n".encode() for index in range(100))

        batches = list(iter_csv_batches(io.BytesIO(text), header=_HEADER, context=_CONTEXT, path=_PATH, block_bytes=64))

        assert len(batches) > 10
        assert [row["id"] for batch in batches for row in _rows(batch)] == [str(index) for index in range(100)]

    def test_default_block_size(self, mocker: MockerFixture) -> None:
        reader = mocker.spy(_batches.pacsv, "open_csv")

        list(iter_csv_batches(io.BytesIO(_TEXT), header=_HEADER, context=_CONTEXT, path=_PATH))

        assert reader.call_args.kwargs["read_options"].block_size == _batches.CSV_BATCH_BYTES == 1024 * 1024

    def test_reads_a_caller_owned_gzip_stream(self) -> None:
        with gzip.GzipFile(fileobj=io.BytesIO(gzip.compress(_TEXT, mtime=0)), mode="rb") as source:
            rows = [
                row
                for batch in iter_csv_batches(source, header=_HEADER, context=_CONTEXT, path=_PATH)
                for row in _rows(batch)
            ]
            assert not source.closed
        assert rows == _EXPECTED

    @pytest.mark.parametrize("payload", [b"id,gene,x\n", b"\xef\xbb\xbfid,gene,x\n"])
    def test_header_only(self, payload: bytes) -> None:
        batches = list(iter_csv_batches(io.BytesIO(payload), header=_HEADER, context=_CONTEXT, path=_PATH))

        assert sum(batch.num_rows for batch in batches) == 0

    @pytest.mark.parametrize("record", [b"1,A\n", b"1,A,0,extra\n"])
    def test_malformed_record_fails_when_its_batch_is_reached(self, record: bytes) -> None:
        text = b"id,gene,x\n" + b"0,A,0\n" * 40 + record
        batches = iter_csv_batches(io.BytesIO(text), header=_HEADER, context=_CONTEXT, path=_PATH, block_bytes=64)

        assert next(batches).num_rows > 0
        with pytest.raises(ReaderFormatError) as caught:
            list(batches)

        assert caught.value.found.startswith("malformed CSV records (CSV parse error")
        assert caught.value.expected == "records with one field per header column"
        assert caught.value.context == _CONTEXT
        assert caught.value.path == _PATH
        assert isinstance(caught.value.__cause__, pa.ArrowInvalid)

    def test_record_longer_than_a_block_fails(self) -> None:
        text = b"id,gene,x\n1," + b"G" * 200 + b",0\n"

        with pytest.raises(ReaderFormatError, match="malformed CSV records"):
            list(iter_csv_batches(io.BytesIO(text), header=_HEADER, context=_CONTEXT, path=_PATH, block_bytes=64))

    @pytest.mark.parametrize("outcome", ["complete", "malformed", "abandoned"])
    def test_arrow_reader_closes_on_every_outcome(self, mocker: MockerFixture, outcome: str) -> None:
        actual_open = pacsv.open_csv
        readers: list[MagicMock] = []

        def track(source: io.BytesIO, **options: object) -> MagicMock:
            reader = actual_open(source, **options)
            tracked = mocker.MagicMock(wraps=reader)
            tracked.__enter__.return_value = reader
            tracked.__exit__.side_effect = reader.__exit__
            readers.append(tracked)
            return tracked

        mocker.patch.object(_batches.pacsv, "open_csv", side_effect=track)
        text = b"id,gene,x\n" + b"0,A,0\n" * 40 + (b"1,A\n" if outcome == "malformed" else b"")
        batches = iter_csv_batches(io.BytesIO(text), header=_HEADER, context=_CONTEXT, path=_PATH, block_bytes=64)
        if outcome == "complete":
            assert sum(batch.num_rows for batch in batches) == 40
        elif outcome == "malformed":
            with pytest.raises(ReaderFormatError):
                list(batches)
        else:
            next(batches)
            batches.close()

        assert len(readers) == 1
        readers[0].__exit__.assert_called_once()


class TestReadCsvTable:
    def test_reads_every_column_as_exact_text(self) -> None:
        source = io.BytesIO(_TEXT)

        table = read_csv_table(source, header=_HEADER, context=_CONTEXT, path=_PATH)

        assert table.schema == pa.schema([(name, pa.string()) for name in _HEADER])
        assert _rows(table) == _EXPECTED
        assert not source.closed

    def test_projects_columns_in_the_requested_order(self) -> None:
        table = read_csv_table(io.BytesIO(_TEXT), header=_HEADER, context=_CONTEXT, path=_PATH, columns=("x", "id"))

        assert table.column_names == ["x", "id"]
        assert table.column("id").to_pylist() == ["9007199254740993", "0012", "3"]

    def test_header_only(self) -> None:
        table = read_csv_table(io.BytesIO(b"id,gene,x\n"), header=_HEADER, context=_CONTEXT, path=_PATH)

        assert table.num_rows == 0
        assert table.schema == pa.schema([(name, pa.string()) for name in _HEADER])

    @pytest.mark.parametrize("record", [b"1,A\n", b"1,A,0,extra\n", b'1,"A,0\n'])
    def test_unprojected_record_width_is_checked(self, record: bytes) -> None:
        with pytest.raises(ReaderFormatError, match=r"malformed CSV records \(") as caught:
            read_csv_table(
                io.BytesIO(b"id,gene,x\n0,A,0\n" + record),
                header=_HEADER,
                context=_CONTEXT,
                path=_PATH,
                columns=("id",),
            )

        assert caught.value.context == _CONTEXT
        assert caught.value.path == _PATH

    def test_parses_like_the_lazy_block_reader(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "counts.csv")
        text = b'id,gene,x\n9007199254740993,"A,B",\n0012,"say ""hi""",0.10000000000000002\n'
        path.write_bytes(text)

        eager = read_csv_table(io.BytesIO(text), header=_HEADER, context=_CONTEXT, path=path, columns=("gene", "x"))
        lazy = read_csv_blocks(path, header=_HEADER, columns=("gene", "x"), context=_CONTEXT).compute()

        assert _rows(eager) == lazy.to_dict(orient="records")

    def test_arrow_quoting_limitation_is_documented_behaviour(self) -> None:
        # A quote inside an unquoted field is literal text, as Arrow parses it.
        table = read_csv_table(io.BytesIO(b'id,gene,x\n1,A"B,0\n'), header=_HEADER, context=_CONTEXT, path=_PATH)

        assert table.column("gene").to_pylist() == ['A"B']
