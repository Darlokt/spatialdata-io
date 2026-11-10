"""Lazy newline-aligned CSV block partitions parsed by Arrow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cloudpickle
import dask
import pandas as pd
import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.csv import _blocks, read_csv_blocks
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile

if TYPE_CHECKING:
    from pathlib import Path

    import dask.dataframe as dd
    from pytest_mock import MockerFixture

_CONTEXT = ReaderErrorContext("test", "transcripts")
_HEADER = ("id", "gene", "x")


def _file(tmp_path: Path, payload: bytes, name: str = "transcripts.csv") -> ArtifactFile:
    path = ArtifactFile(tmp_path / name)
    path.write_bytes(payload)
    return path


def _partitions(frame: dd.DataFrame) -> list[pd.DataFrame]:
    return list(dask.compute(*frame.to_delayed(), scheduler="synchronous"))


class TestReadCsvBlocks:
    # Arrow's typed CSV grammar, the numeric contract every reader documents.
    @pytest.mark.parametrize(
        ("token", "kind", "value"),
        [
            (b"18446744073709551615", pa.uint64(), 2**64 - 1),
            (b" 0x10\t", pa.uint64(), 16),
            (b"0xffffffffffffffff", pa.int64(), -1),
            (b"0x8000000000000000", pa.int64(), -(2**63)),
            (b"0.10000000000000002", pa.float64(), 0.10000000000000002),
            (b"9007199254740993", pa.float64(), 9007199254740992.0),
            (b"1e-400", pa.float64(), 0.0),
            (b"NA", pa.string(), "NA"),
            (b"", pa.string(), ""),
        ],
    )
    def test_arrow_grammar_parses_declared_types(
        self, tmp_path: Path, token: bytes, kind: pa.DataType, value: object
    ) -> None:
        path = _file(tmp_path, b"id,gene,x\n" + token + b",A,0\n")

        frame = read_csv_blocks(
            path, header=_HEADER, columns=("id", "gene"), context=_CONTEXT, column_types={"id": kind}
        )
        result = frame.compute()

        assert frame.dtypes.tolist() == result.dtypes.tolist() == [pd.ArrowDtype(kind), pd.ArrowDtype(pa.string())]
        assert result["id"].tolist() == [value]
        assert not result.isna().any(axis=None)

    @pytest.mark.parametrize(
        ("token", "kind"),
        [
            (b"", pa.uint64()),
            (b"NA", pa.uint64()),
            (b"-1", pa.uint64()),
            (b"18446744073709551616", pa.uint64()),
            (b"0x10000000000000000", pa.uint64()),
            (b"9223372036854775808", pa.int64()),
            (b"2.0", pa.uint64()),
            (b"+1", pa.uint64()),
            (b"1_0", pa.uint64()),
        ],
    )
    def test_arrow_grammar_rejects_only_when_computed(self, tmp_path: Path, token: bytes, kind: pa.DataType) -> None:
        path = _file(tmp_path, b"id,gene,x\n" + token + b",A,0\n")
        frame = read_csv_blocks(path, header=_HEADER, columns=("id",), context=_CONTEXT, column_types={"id": kind})

        with pytest.raises(ReaderFormatError) as caught:
            frame.compute()

        assert caught.value.context == _CONTEXT
        assert caught.value.path == path
        assert isinstance(caught.value.__cause__, pa.ArrowInvalid)

    @pytest.mark.parametrize("payload", [b"id,gene,x\n", b"id,gene,x"])
    def test_header_only_file_has_one_empty_typed_partition(self, tmp_path: Path, payload: bytes) -> None:
        path = _file(tmp_path, payload)
        types = {"id": pa.uint64(), "x": pa.float64()}

        frame = read_csv_blocks(path, header=_HEADER, columns=tuple(types), context=_CONTEXT, column_types=types)

        result = frame.compute()
        assert frame.npartitions == 1
        assert result.empty
        assert result.dtypes.tolist() == frame.dtypes.tolist() == [pd.ArrowDtype(dtype) for dtype in types.values()]

    def test_quoting_line_endings_and_projection_order(self, tmp_path: Path) -> None:
        path = _file(tmp_path, b'\xef\xbb\xbfid,gene,x\r\n001,"say ""hi""",0\r\n002,"A,B",1')

        result = read_csv_blocks(path, header=_HEADER, columns=("gene", "id"), context=_CONTEXT).compute()

        assert list(result.columns) == ["gene", "id"]
        assert result["gene"].tolist() == ['say "hi"', "A,B"]
        assert result["id"].tolist() == ["001", "002"]

    @pytest.mark.parametrize(
        ("record", "lines"),
        [
            (b'1,"two\nlines",0\n', ('1,"two', 'lines",0')),
            (b'1,5" tall,0\n', ('1,5" tall,0',)),
            (b'1,"""",0\n2,A"\n', ('2,A"',)),
        ],
    )
    def test_odd_quotes_fail_at_every_block_size(self, tmp_path: Path, record: bytes, lines: tuple[str, ...]) -> None:
        path = _file(tmp_path, b"id,gene,x\n" + b"0,A,0\n" * 8 + record + b"3,C,2\n" * 8)

        for block_bytes in range(4, path.stat().st_size + 2):
            frame = read_csv_blocks(path, header=_HEADER, columns=("id",), context=_CONTEXT, block_bytes=block_bytes)
            with pytest.raises(ReaderFormatError, match="odd number of double quotes") as raised:
                frame.compute(scheduler="synchronous")
            assert raised.value.context == _CONTEXT
            assert raised.value.path == path
            assert any(repr(line) in str(raised.value) for line in lines)

    def test_paired_quotes_give_the_same_records_at_every_block_size(self, tmp_path: Path) -> None:
        rows = [f'{index},"G,{index} ""q""",{index}.5\n'.encode() for index in range(12)]
        path = _file(tmp_path, b"id,gene,x\n" + b"".join(rows))
        expected = [f'G,{index} "q"' for index in range(12)]

        for block_bytes in range(4, path.stat().st_size + 2):
            frame = read_csv_blocks(path, header=_HEADER, columns=("gene",), context=_CONTEXT, block_bytes=block_bytes)
            assert frame.compute(scheduler="synchronous")["gene"].tolist() == expected

    def test_construction_reads_no_record_and_serializes(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _file(tmp_path, b"id,gene,x\n1,A,0.5\n")
        parser = mocker.spy(_blocks, "_read_block")

        frame = read_csv_blocks(path, header=_HEADER, columns=("gene", "x"), context=_CONTEXT)

        parser.assert_not_called()
        cloudpickle.dumps(frame)
        assert list(frame.columns) == ["gene", "x"]
        assert frame.dtypes.tolist() == [pd.ArrowDtype(pa.string())] * 2

    # A 10-byte header line and 100 records of 12 bytes: max(1, 1210 // block_bytes) blocks.
    @pytest.mark.parametrize(("block_bytes", "npartitions"), [(2000, 1), (1210, 1), (605, 2), (121, 10), (64, 18)])
    def test_block_size_sets_the_partition_count_without_changing_records(
        self, tmp_path: Path, block_bytes: int, npartitions: int
    ) -> None:
        rows = [f"{index:03d},G{index:02d},{index % 7}.5\n".encode() for index in range(100)]
        path = _file(tmp_path, b"id,gene,x\n" + b"".join(rows))
        types = {"x": pa.float64()}
        whole = read_csv_blocks(path, header=_HEADER, columns=_HEADER, context=_CONTEXT, column_types=types)

        frame = read_csv_blocks(
            path, header=_HEADER, columns=_HEADER, context=_CONTEXT, column_types=types, block_bytes=block_bytes
        )
        partitions = _partitions(frame)

        assert path.stat().st_size == 1210
        assert frame.npartitions == len(partitions) == npartitions
        assert all(partition.index.tolist() == list(range(len(partition))) for partition in partitions)
        combined = pd.concat(partitions, ignore_index=True)
        pd.testing.assert_frame_equal(combined, whole.compute(), check_exact=True)
        assert combined["id"].tolist() == [f"{index:03d}" for index in range(100)]

    def test_record_longer_than_a_block_leaves_empty_partitions(self, tmp_path: Path) -> None:
        path = _file(tmp_path, b"id,gene,x\n1,A,0\n2," + b"G" * 100 + b",1\n3,C,2\n")

        frame = read_csv_blocks(path, header=_HEADER, columns=("id",), context=_CONTEXT, block_bytes=16)
        partitions = _partitions(frame)

        assert any(partition.empty for partition in partitions)
        assert pd.concat(partitions)["id"].tolist() == ["1", "2", "3"]

    @pytest.mark.parametrize(("length", "readable"), [(1024 * 1024 - 8, True), (2 * 1024 * 1024 + 1, False)])
    def test_records_beyond_two_arrow_blocks_fail(self, tmp_path: Path, *, length: int, readable: bool) -> None:
        path = _file(tmp_path, b"id,gene,x\n" + b"0,A,0\n" * 3 + b"1," + b"G" * length + b",1\n")
        frame = read_csv_blocks(path, header=_HEADER, columns=("id",), context=_CONTEXT, block_bytes=2**30)

        if readable:
            assert frame.compute()["id"].tolist() == ["0", "0", "0", "1"]
        else:
            with pytest.raises(ReaderFormatError, match="straddles two block boundaries"):
                frame.compute()

    @pytest.mark.parametrize("record", [b"1,A\n", b"1,A,0,extra\n"])
    def test_unprojected_record_width_errors_only_when_its_partition_runs(self, tmp_path: Path, record: bytes) -> None:
        path = _file(tmp_path, b"id,gene,x\n" + b"0,A,0\n" * 4 + record)
        frame = read_csv_blocks(path, header=_HEADER, columns=("gene",), context=_CONTEXT, block_bytes=16)

        first = frame.partitions[0].compute()["gene"].tolist()
        assert first
        assert set(first) == {"A"}
        with pytest.raises(ReaderFormatError, match="malformed CSV") as raised:
            frame.partitions[-1].compute()

        assert raised.value.context == _CONTEXT
        assert raised.value.path == path

    def test_literal_glob_characters_in_the_path(self, tmp_path: Path) -> None:
        directory = tmp_path / "[roi]*?"
        directory.mkdir()
        path = _file(directory, b"id,gene,x\n1,A,0\n")

        result = read_csv_blocks(path, header=_HEADER, columns=("gene",), context=_CONTEXT).compute()

        assert result["gene"].tolist() == ["A"]


class TestPairedQuotes:
    @pytest.mark.parametrize(
        "block",
        [
            b"",
            b"a,b\n",
            b'"a",b\n"c"\n',
            b'"a\nb"\n',
            b'x\n\n"y""z",1\r\n',
            b'"\n"\n',
            b'a"b"c"\nd"',
            b'"x"\n"y',
            b'\n"""\n"\n',
        ],
    )
    @pytest.mark.parametrize("slice_bytes", [1, 2, 3, 5, 1024 * 1024])
    def test_matches_per_line_quote_parity_across_slices(
        self, mocker: MockerFixture, block: bytes, slice_bytes: int
    ) -> None:
        mocker.patch.object(_blocks, "_QUOTE_SLICE_BYTES", slice_bytes)

        expected = all(line.count(b'"') % 2 == 0 for line in block.split(b"\n"))

        assert _blocks._paired_quotes(block) is expected


class TestReadBlock:
    @pytest.mark.parametrize("record", [b"1,A,0\n", b"1,A\n", b"1,\xff,0\n"])
    def test_owned_buffer_closes_on_success_and_parse_failure(
        self, tmp_path: Path, mocker: MockerFixture, record: bytes
    ) -> None:
        owner = pa.BufferReader(record)
        mocker.patch.object(_blocks.pa, "BufferReader", return_value=owner)
        schema = pa.schema([(name, pa.string()) for name in _HEADER])
        spec = _blocks._BlockSpec(ArtifactFile(tmp_path / "transcripts.csv"), _HEADER, schema, _CONTEXT)

        if record == b"1,A,0\n":
            assert _blocks._read_block(record, spec, first=False)["gene"].tolist() == ["A"]
        else:
            with pytest.raises(ReaderFormatError):
                _blocks._read_block(record, spec, first=False)

        assert owner.closed
