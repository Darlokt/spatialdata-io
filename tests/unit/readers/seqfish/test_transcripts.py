"""Bounded transcript metadata, lazy partition validation and public point parsing."""

from __future__ import annotations

import pickle
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from dask.callbacks import Callback

import spatialdata_io.readers.seqfish._transcripts as transcript_module
from spatialdata_io.readers._utils.csv import _blocks as csv_blocks
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
            "name,x,y,cell\n" + "".join(f"GeneB,{row},2,{cell}\n" for row, cell in enumerate(assignments)),
            encoding="utf-8",
        )
        mocker.patch.object(csv_blocks, "CSV_BLOCK_BYTES", 96)
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
        assert result["x"].tolist() == list(range(len(assignments)))

    @pytest.mark.parametrize("invalid_cell", ["-1", str(2**64), "1.1", "NaN"])
    def test_invalid_later_assignment_is_checked_when_executed(
        self, tmp_path: Path, mocker: MockerFixture, invalid_cell: str
    ) -> None:
        path = _write_transcripts(tmp_path / "transcripts.csv")
        with path.open("a", encoding="utf-8") as output:
            output.write(f"annotation,2.5,GeneB,{invalid_cell},1,1.5\n")
        mocker.patch.object(csv_blocks, "CSV_BLOCK_BYTES", 128)
        points = read_transcripts(path, features=_FEATURES, frame="Roi1")
        assert not points.partitions[0].compute().empty
        with pytest.raises(ReaderFormatError, match="identifier"):
            points.partitions[-1].compute()

    def test_projected_known_categories_construct_without_tasks_and_serialize(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        path = _write_transcripts(tmp_path / "transcripts.csv")
        mocker.patch.object(csv_blocks, "CSV_BLOCK_BYTES", 128)
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
        [("Unknown", "1.5", "unknown transcript"), ('"Gene,A"', "nan", "finite float64")],
    )
    def test_bad_later_partition_does_not_execute_with_first_partition(
        self, tmp_path: Path, mocker: MockerFixture, last_name: str, last_x: str, message: str
    ) -> None:
        path = _write_transcripts(tmp_path / "transcripts.csv", last_name=last_name, last_x=last_x)
        mocker.patch.object(csv_blocks, "CSV_BLOCK_BYTES", 128)
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
            (b"name,x,x,y\nGeneB,1,2,3\n", "duplicate"),
            (b"name,x\nGeneB,1\n", "name, x and y"),
            (b'"name,x,y\nGeneB,1,2\n', "malformed CSV header"),
            (b"name,\xff,y\n", "malformed CSV header"),
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

    def test_large_first_record_is_not_sampled_at_construction(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = ArtifactFile(tmp_path / "transcripts.csv")
        path.write_text("name,x,y,annotation\nGeneB,1,2," + "a" * (128 * 1024) + "\n", encoding="utf-8")
        storage = mocker.spy(csv_blocks, "read_bytes")
        parser = mocker.spy(csv_blocks, "_read_block")
        points = read_transcripts(path, features=_FEATURES, frame="Roi1")
        parser.assert_not_called()
        assert storage.call_args.kwargs["sample"] is False
        result = points.compute()
        assert result["name"].tolist() == ["GeneB"]
        assert "annotation" not in result

    @pytest.mark.parametrize("record", ["annotation,2.5,GeneB,0,1,1.5,unexpected\n", "annotation,2.5,GeneB,0,1\n"])
    def test_malformed_record_width_is_not_silently_projected_away(
        self, tmp_path: Path, mocker: MockerFixture, record: str
    ) -> None:
        path = _write_transcripts(tmp_path / "transcripts.csv")
        with path.open("a", encoding="utf-8") as output:
            output.write(record)
        mocker.patch.object(csv_blocks, "CSV_BLOCK_BYTES", 128)
        points = read_transcripts(path, features=_FEATURES, frame="Roi1")
        assert not points.partitions[0].compute().empty
        with pytest.raises(ValueError, match=r"CSV|columns|fields|malformed"):
            points.partitions[-1].compute()

    def test_lexical_coordinates_and_large_mixed_numeric_cell_ids_remain_exact(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "transcripts.csv")
        path.write_text(
            "name,x,y,cell\nGeneB,0.10000000000000002,2.5,9007199254740993\nGeneB,1.5,2.5,2.0\n",
            encoding="utf-8",
        )
        points = read_transcripts(path, features=_FEATURES, frame="Roi1")
        result = points.compute()
        assert result["cell"].tolist() == [9007199254740993, 2]
        assert result["x"].iloc[0] == float("0.10000000000000002")

    def test_source_path_glob_characters_are_literal(self, tmp_path: Path) -> None:
        root = tmp_path / "[ROI]*?"
        root.mkdir()
        path = _write_transcripts(root / "transcripts.csv", rows=2)
        result = read_transcripts(path, features=_FEATURES, frame="Roi1").compute()
        assert result["name"].tolist() == ["GeneB", "Gene,A"]
