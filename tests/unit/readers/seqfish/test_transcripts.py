"""Bounded transcript metadata, lazy partition validation and public point parsing."""

from __future__ import annotations

import pickle
import warnings
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from dask.callbacks import Callback

import spatialdata_io.readers.seqfish._transcripts as transcript_module
from spatialdata_io.readers._utils.csv import _blocks as csv_blocks
from spatialdata_io.readers._utils.csv import read_csv_blocks
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.seqfish._transcripts import read_transcripts

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

_FEATURES = ("GeneB", "Gene,A", "ZeroCount")


def _write_transcripts(path: Path, *, rows: int = 40, last_name: str = '"Gene,A"', last_x: str = "1.5") -> ArtifactFile:
    """Include quoted features and an ignored annotation in several small blocks."""
    records = ["ignored,y,name,cell,z,x\n"]
    records.extend(
        f"annotation,2.5,{last_name if i == rows - 1 else 'GeneB'},0,1,{last_x if i == rows - 1 else '1.5'}\n"
        for i in range(rows)
    )
    path.write_text("".join(records), encoding="utf-8")
    return ArtifactFile(path)


class TestReadTranscripts:
    """Execute only selected source partitions and preserve scientific annotations."""

    def test_mixed_uint64_assignments_compute_across_small_partitions(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        assignments = [0, 2, 2**53 + 1, 2**63 + 1, 2**64 - 1] * 12
        path = ArtifactFile(tmp_path / "transcripts.csv")
        path.write_text(
            "name,x,y,cell,z\n" + "".join(f"GeneB,{row},2,{cell},{cell}\n" for row, cell in enumerate(assignments)),
            encoding="utf-8",
        )
        mocker.patch.object(transcript_module, "read_csv_blocks", partial(read_csv_blocks, block_bytes=96))
        tasks = mocker.Mock()
        with Callback(posttask=tasks):
            points = read_transcripts(path, features=_FEATURES, frame="Roi1")
        tasks.assert_not_called()
        assert points.npartitions > 5
        assert points.dtypes["cell"] == pd.ArrowDtype(pa.uint64())
        partitions = [partition.compute() for partition in points.to_delayed()]
        assert all(partition["cell"].dtype == pd.ArrowDtype(pa.uint64()) for partition in partitions)
        result = points.compute()
        assert result["cell"].dtype == pd.ArrowDtype(pa.uint64())
        assert result["cell"].tolist() == assignments
        assert result["z_raw"].dtype == pd.ArrowDtype(pa.uint64())
        assert result["z_raw"].tolist() == assignments
        assert result["x"].tolist() == list(range(len(assignments)))

    @pytest.mark.parametrize(
        ("record", "found"),
        [
            ("annotation,2.5,GeneB,-1,1,1.5\n", "CSV conversion error to uint64: invalid value '-1'"),
            ("annotation,2.5,GeneB,1,-1,1.5\n", "CSV conversion error to uint64: invalid value '-1'"),
            ('"multiline\nannotation",2.5,GeneB,1,1,1.5\n', "odd number of double quotes"),
        ],
        ids=["cell", "plane", "quoted-newline"],
    )
    def test_invalid_later_record_is_checked_when_executed(
        self, tmp_path: Path, mocker: MockerFixture, record: str, found: str
    ) -> None:
        path = _write_transcripts(tmp_path / "transcripts.csv")
        with path.open("a", encoding="utf-8") as output:
            output.write(record)
        mocker.patch.object(transcript_module, "read_csv_blocks", partial(read_csv_blocks, block_bytes=128))
        points = read_transcripts(path, features=_FEATURES, frame="Roi1")
        assert not points.partitions[0].compute().empty
        with pytest.raises(ReaderFormatError) as caught:
            points.partitions[-1].compute()
        assert found in caught.value.found
        assert caught.value.context.artifact == "TranscriptList"

    def test_projected_known_categories_construct_without_tasks_and_serialize(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        path = _write_transcripts(tmp_path / "transcripts.csv")
        mocker.patch.object(transcript_module, "read_csv_blocks", partial(read_csv_blocks, block_bytes=128))
        tasks = mocker.Mock()
        filters = list(warnings.filters)
        with Callback(posttask=tasks):
            points = read_transcripts(path, features=_FEATURES, frame="Roi1")
        tasks.assert_not_called()
        assert warnings.filters == filters
        assert points.npartitions > 1
        assert not points.known_divisions
        assert set(points.columns) == {"x", "y", "name", "cell", "z_raw"}
        assert points["name"].cat.known
        assert points["name"].cat.categories.tolist() == list(_FEATURES)
        assert points.dtypes["x"] == np.dtype(np.float64)
        assert points.dtypes["y"] == np.dtype(np.float64)
        restored = pickle.loads(pickle.dumps(points))  # noqa: S301, RUF100 - Test-owned path-backed graph.
        result = restored.compute()
        assert len(result) == 40
        assert result["name"].tolist() == ["GeneB"] * 39 + ["Gene,A"]
        np.testing.assert_array_equal(result[["x", "y"]].to_numpy(), np.tile([1.5, 2.5], (40, 1)))
        assert result["cell"].tolist() == [0] * 40
        assert result["z_raw"].tolist() == [1] * 40
        assert result.index.duplicated().any()

    def test_optional_annotations_are_absent_and_empty_points_are_supported(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "transcripts.csv")
        path.write_text("name,x,y\n", encoding="utf-8")
        points = read_transcripts(path, features=_FEATURES, frame="Roi1")
        assert set(points.columns) == {"name", "x", "y"}
        assert points.compute().empty

    @pytest.mark.parametrize(
        ("last_name", "last_x", "message"),
        [
            ("Unknown", "1.5", "unknown transcript"),
            ('"Gene,A"', "nan", "nonfinite 'x'"),
            ('"Gene,A"', "-inf", "nonfinite 'x'"),
        ],
    )
    def test_bad_later_partition_does_not_execute_with_first_partition(
        self, tmp_path: Path, mocker: MockerFixture, last_name: str, last_x: str, message: str
    ) -> None:
        path = _write_transcripts(tmp_path / "transcripts.csv", last_name=last_name, last_x=last_x)
        mocker.patch.object(transcript_module, "read_csv_blocks", partial(read_csv_blocks, block_bytes=128))
        points = read_transcripts(path, features=_FEATURES, frame="Roi1")
        assert points.npartitions > 1
        first = points.partitions[0].compute()
        assert set(first["name"]) == {"GeneB"}
        with pytest.raises(ReaderFormatError, match=message) as caught:
            points.partitions[-1].compute()
        assert caught.value.path == path

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            (b"name,x,x,y\nGeneB,1,2,3\n", "repeated column names"),
            (b"name,x\nGeneB,1\n", "name, x and y"),
        ],
    )
    def test_malformed_header_fails_before_dask_parser(
        self, tmp_path: Path, mocker: MockerFixture, content: bytes, message: str
    ) -> None:
        path = ArtifactFile(tmp_path / "transcripts.csv")
        path.write_bytes(content)
        parser = mocker.spy(csv_blocks, "read_bytes")
        with pytest.raises(ReaderFormatError, match=message):
            read_transcripts(path, features=_FEATURES, frame="Roi1")
        parser.assert_not_called()

    def test_warning_filters_restore_when_public_model_parser_fails(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        path = _write_transcripts(tmp_path / "transcripts.csv")
        parser = mocker.patch.object(transcript_module.PointsModel, "parse", side_effect=RuntimeError("injected"))
        initial = list(warnings.filters)
        with pytest.raises(RuntimeError, match="injected"):
            read_transcripts(path, features=_FEATURES, frame="Roi1")
        assert warnings.filters == initial
        parser.assert_called_once()
        assert parser.call_args.kwargs["sort"] is False
