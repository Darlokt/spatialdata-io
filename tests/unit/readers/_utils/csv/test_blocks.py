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
    @pytest.mark.parametrize("record", [b"18446744073709551615,NA,0.10000000000000002\n", b"0,,1e-400\n"])
    def test_native_types_preserve_exact_ids_rounding_and_literal_text(self, tmp_path: Path, record: bytes) -> None:
        path = _file(tmp_path, b"id,gene,x\n" + record)
        types = {"id": pa.uint64(), "gene": pa.string(), "x": pa.float64()}

        frame = read_csv_blocks(path, header=_HEADER, columns=_HEADER, context=_CONTEXT, column_types=types)
        result = frame.compute()

        assert frame.dtypes.tolist() == result.dtypes.tolist() == [pd.ArrowDtype(dtype) for dtype in types.values()]
        if record.startswith(b"0,"):
            assert result.iloc[0].tolist() == [0, "", 0.0]
        else:
            assert result.iloc[0].tolist() == [2**64 - 1, "NA", 0.10000000000000002]
        assert not result.isna().any(axis=None)

    @pytest.mark.parametrize("token", [b"", b"NA", b"-1", b"18446744073709551616"])
    def test_invalid_unsigned_values_fail_only_when_computed(self, tmp_path: Path, token: bytes) -> None:
        path = _file(tmp_path, b"id,gene,x\n" + token + b",A,0\n")
        frame = read_csv_blocks(
            path, header=_HEADER, columns=("id",), context=_CONTEXT, column_types={"id": pa.uint64()}
        )

        with pytest.raises(ReaderFormatError) as caught:
            frame.compute()

        assert caught.value.context == _CONTEXT
        assert caught.value.path == path
        assert isinstance(caught.value.__cause__, pa.ArrowInvalid)

    def test_empty_partition_keeps_declared_types(self, tmp_path: Path) -> None:
        path = _file(tmp_path, b"id,gene,x\n")
        types = {"id": pa.uint64(), "x": pa.float64()}

        frame = read_csv_blocks(path, header=_HEADER, columns=tuple(types), context=_CONTEXT, column_types=types)

        result = frame.compute()
        assert result.empty
        assert result.dtypes.tolist() == frame.dtypes.tolist() == [pd.ArrowDtype(dtype) for dtype in types.values()]

    def test_native_quoting_and_source_line_endings(self, tmp_path: Path) -> None:
        path = _file(tmp_path, b'\xef\xbb\xbfid,gene,x\r\n001,"say ""hi""",0\r\n002,A"B,1')

        result = read_csv_blocks(path, header=_HEADER, columns=("gene", "id"), context=_CONTEXT).compute()

        assert list(result.columns) == ["gene", "id"]
        assert result["gene"].tolist() == ['say "hi"', 'A"B']
        assert result["id"].tolist() == ["001", "002"]

    def test_construction_reads_no_record_and_serializes(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _file(tmp_path, b"id,gene,x\n1,A,0.5\n")
        parser = mocker.spy(_blocks, "_read_block")

        frame = read_csv_blocks(path, header=_HEADER, columns=("gene", "x"), context=_CONTEXT)

        parser.assert_not_called()
        cloudpickle.dumps(frame)
        assert list(frame.columns) == ["gene", "x"]
        assert frame.dtypes.tolist() == [pd.ArrowDtype(pa.string())] * 2

    def test_partitions_follow_newline_aligned_blocks_in_file_order(self, tmp_path: Path) -> None:
        rows = [f"{index},G{index},{index}.5\n".encode() for index in range(6)]
        path = _file(tmp_path, b"id,gene,x\n" + b"".join(rows))

        frame = read_csv_blocks(path, header=_HEADER, columns=("id", "gene"), context=_CONTEXT, block_bytes=24)
        partitions = _partitions(frame)

        assert frame.npartitions == len(partitions) > 1
        assert all(partition.index.tolist() == list(range(len(partition))) for partition in partitions)
        combined = pd.concat(partitions, ignore_index=True)
        assert combined["id"].tolist() == [str(index) for index in range(6)]
        assert combined["gene"].tolist() == [f"G{index}" for index in range(6)]

    def test_default_block_size_is_the_shared_constant(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _file(tmp_path, b"id,gene,x\n1,A,0\n")
        blocks = mocker.spy(_blocks, "read_bytes")

        read_csv_blocks(path, header=_HEADER, columns=("gene",), context=_CONTEXT)

        assert blocks.call_args.kwargs["blocksize"] == _blocks.CSV_BLOCK_BYTES == 8 * 1024 * 1024

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
        pd.testing.assert_frame_equal(pd.concat(partitions, ignore_index=True), whole.compute(), check_exact=True)

    def test_keeps_quoted_delimiters_empty_strings_and_exact_text(self, tmp_path: Path) -> None:
        path = _file(tmp_path, b'id,gene,x\n9007199254740993,"A,B",\n0012,"",0.10000000000000002\n')

        result = read_csv_blocks(path, header=_HEADER, columns=_HEADER, context=_CONTEXT).compute()

        assert result["id"].tolist() == ["9007199254740993", "0012"]
        assert result["gene"].tolist() == ["A,B", ""]
        assert result["x"].tolist() == ["", "0.10000000000000002"]
        assert not result.isna().any(axis=None)

    def test_applies_caller_types_to_projected_columns(self, tmp_path: Path) -> None:
        path = _file(tmp_path, b"id,gene,x\n9007199254740993,A,1.5\n")

        result = read_csv_blocks(
            path, header=_HEADER, columns=("id", "x"), context=_CONTEXT, column_types={"id": pa.uint64()}
        ).compute()

        assert result.dtypes.tolist() == [pd.ArrowDtype(pa.uint64()), pd.ArrowDtype(pa.string())]
        assert result["id"].tolist() == [9007199254740993]

    @pytest.mark.parametrize("payload", [b"id,gene,x\n", b"id,gene,x"])
    def test_header_only_file_has_one_empty_partition(self, tmp_path: Path, payload: bytes) -> None:
        path = _file(tmp_path, payload)

        result = read_csv_blocks(path, header=_HEADER, columns=("gene",), context=_CONTEXT).compute()

        assert result.empty
        assert list(result.columns) == ["gene"]

    def test_record_longer_than_a_block_leaves_empty_partitions(self, tmp_path: Path) -> None:
        path = _file(tmp_path, b"id,gene,x\n1,A,0\n2," + b"G" * 100 + b",1\n3,C,2\n")

        frame = read_csv_blocks(path, header=_HEADER, columns=("id",), context=_CONTEXT, block_bytes=16)
        partitions = _partitions(frame)

        assert any(partition.empty for partition in partitions)
        assert pd.concat(partitions)["id"].tolist() == ["1", "2", "3"]

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
