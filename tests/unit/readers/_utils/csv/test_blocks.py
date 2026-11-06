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
    def test_construction_reads_no_record_and_serializes(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _file(tmp_path, b"id,gene,x\n1,A,0.5\n")
        parser = mocker.spy(_blocks, "_read_block")

        frame = read_csv_blocks(path, header=_HEADER, columns=("gene", "x"), context=_CONTEXT)

        parser.assert_not_called()
        cloudpickle.dumps(frame)
        assert list(frame.columns) == ["gene", "x"]
        assert frame.dtypes.tolist() == [pd.ArrowDtype(pa.string())] * 2

    def test_partitions_follow_newline_aligned_blocks_in_file_order(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        rows = [f"{index},G{index},{index}.5\n".encode() for index in range(6)]
        path = _file(tmp_path, b"id,gene,x\n" + b"".join(rows))
        mocker.patch.object(_blocks, "CSV_BLOCK_BYTES", 24)

        frame = read_csv_blocks(path, header=_HEADER, columns=("id", "gene"), context=_CONTEXT)
        partitions = _partitions(frame)

        assert frame.npartitions == len(partitions) > 1
        assert all(partition.index.tolist() == list(range(len(partition))) for partition in partitions)
        combined = pd.concat(partitions, ignore_index=True)
        assert combined["id"].tolist() == [str(index) for index in range(6)]
        assert combined["gene"].tolist() == [f"G{index}" for index in range(6)]

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

    def test_record_longer_than_a_block_leaves_empty_partitions(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _file(tmp_path, b"id,gene,x\n1,A,0\n2," + b"G" * 100 + b",1\n3,C,2\n")
        mocker.patch.object(_blocks, "CSV_BLOCK_BYTES", 16)

        frame = read_csv_blocks(path, header=_HEADER, columns=("id",), context=_CONTEXT)
        partitions = _partitions(frame)

        assert any(partition.empty for partition in partitions)
        assert pd.concat(partitions)["id"].tolist() == ["1", "2", "3"]

    @pytest.mark.parametrize("record", [b"1,A\n", b"1,A,0,extra\n"])
    def test_unprojected_record_width_errors_only_when_its_partition_runs(
        self, tmp_path: Path, mocker: MockerFixture, record: bytes
    ) -> None:
        path = _file(tmp_path, b"id,gene,x\n" + b"0,A,0\n" * 4 + record)
        mocker.patch.object(_blocks, "CSV_BLOCK_BYTES", 16)
        frame = read_csv_blocks(path, header=_HEADER, columns=("gene",), context=_CONTEXT)

        first = frame.partitions[0].compute()["gene"].tolist()
        assert first
        assert set(first) == {"A"}
        with pytest.raises(ReaderFormatError, match="malformed CSV records") as raised:
            frame.partitions[-1].compute()

        assert raised.value.context == _CONTEXT
        assert raised.value.path == path

    def test_literal_glob_characters_in_the_path(self, tmp_path: Path) -> None:
        directory = tmp_path / "[roi]*?"
        directory.mkdir()
        path = _file(directory, b"id,gene,x\n1,A,0\n")

        result = read_csv_blocks(path, header=_HEADER, columns=("gene",), context=_CONTEXT).compute()

        assert result["gene"].tolist() == ["A"]
