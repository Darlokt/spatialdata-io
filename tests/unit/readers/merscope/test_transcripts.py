"""Projected MERSCOPE source blocks and deferred public point construction."""

from __future__ import annotations

import pickle
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from dask.callbacks import Callback
from numpy.testing import assert_array_equal
from spatialdata.transformations import Affine, get_transformation

import spatialdata_io.readers.merscope._transcripts as transcript_module
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.merscope._layout import TranscriptSource
from spatialdata_io.readers.merscope._transcripts import read_transcripts

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

_FEATURES = ("GeneB", "Gene,A", "Undetected")
_TRANSFORM = Affine(
    np.array([[2, 0.25, 10], [0, 3, -7], [0, 0, 1]], dtype=np.float64),
    input_axes=("x", "y"),
    output_axes=("x", "y"),
)


def _write_source(
    path: Path,
    *,
    assignment: Literal["current", "original"] = "current",
    last_gene: str = '"Gene,A"',
    last_cell: str = "-1",
) -> TranscriptSource:
    """Write quoted genes, an ignored source index, and projected annotations."""
    records = [",gene,global_x,global_y,global_z,cell_id,fov,barcode_id,transcript_id,x,y,ignored\n"]
    records.extend(f"{i},GeneB,1.5,2.5,2,9007199254740993,0,1,ENST.1,100,200,annotation\n" for i in range(39))
    records.append(f"39,{last_gene},1.5,2.5,2,{last_cell},0,1,ENST.1,100,200,annotation\n")
    path.write_text("".join(records), encoding="utf-8")
    vocabulary = ArtifactFile(path.parent / "cell_by_gene.csv")
    vocabulary.write_text('cell,GeneB,"Gene,A",Undetected\n', encoding="utf-8")
    return TranscriptSource(ArtifactFile(path), vocabulary, assignment)


class TestReadTranscripts:
    """Check physical XY, exact identities, projection and path-backed execution."""

    def test_constructs_known_categories_without_tasks_and_serializes(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        source = _write_source(tmp_path / "transcripts.csv")
        mocker.patch.object(transcript_module, "_BLOCK_BYTES", 256)
        tasks = mocker.Mock()
        initial_filters = list(warnings.filters)
        with Callback(posttask=tasks):
            points = read_transcripts(
                source, features=_FEATURES, transform=_TRANSFORM, allowed_cell_ids=("9007199254740993",)
            )

        tasks.assert_not_called()
        assert warnings.filters == initial_filters
        assert points.npartitions > 1
        assert points["gene"].cat.known
        assert points["gene"].cat.categories.tolist() == list(_FEATURES)
        assert points.dtypes["x"] == np.dtype(np.float64)
        assert points.dtypes["y"] == np.dtype(np.float64)
        assert set(points.columns) == {"gene", "x", "y", "z_raw", "cell_id", "fov", "barcode_id", "transcript_id"}
        assert get_transformation(points, to_coordinate_system="global") == _TRANSFORM

        restored = pickle.loads(pickle.dumps(points))  # noqa: S301, RUF100 - Test-owned path-backed graph.
        result = restored.compute()
        assert result["gene"].tolist() == ["GeneB"] * 39 + ["Gene,A"]
        assert result["cell_id"].tolist() == ["9007199254740993"] * 39 + ["-1"]
        assert result["transcript_id"].tolist() == ["ENST.1"] * 40
        assert_array_equal(result[["x", "y"]], np.tile([1.5, 2.5], (40, 1)))
        assert result["z_raw"].tolist() == [2] * 40

    def test_original_assignments_are_explicit_annotations(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path / "transcripts.csv", assignment="original")
        points = read_transcripts(source, features=_FEATURES, transform=_TRANSFORM)
        result = points.compute()
        assert "cell_id" not in result
        assert result["original_cell_id"].tolist() == ["9007199254740993"] * 39 + ["-1"]

    def test_header_only_input_has_typed_empty_points_without_optional_annotations(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path / "transcripts.csv")
        source.path.write_text("gene,global_x,global_y\n", encoding="utf-8")
        points = read_transcripts(source, features=_FEATURES, transform=_TRANSFORM)
        assert set(points.columns) == {"gene", "x", "y"}
        assert points.compute().empty

    @pytest.mark.parametrize(
        ("gene", "cell", "message"), [("Unknown", "-1", "unknown gene"), ("GeneB", "9002", "unknown assigned")]
    )
    def test_invalid_later_partition_does_not_execute_with_first_partition(
        self, tmp_path: Path, mocker: MockerFixture, gene: str, cell: str, message: str
    ) -> None:
        source = _write_source(tmp_path / "transcripts.csv", last_gene=gene, last_cell=cell)
        mocker.patch.object(transcript_module, "_BLOCK_BYTES", 256)
        points = read_transcripts(
            source, features=_FEATURES, transform=_TRANSFORM, allowed_cell_ids=("9007199254740993",)
        )
        assert points.npartitions > 1
        first = points.partitions[0].compute()
        assert set(first["gene"]) == {"GeneB"}
        with pytest.raises(ReaderFormatError, match=message) as caught:
            points.partitions[-1].compute()
        assert caught.value.path == source.path

    @pytest.mark.parametrize("content", [b'"gene,global_x,global_y\n', b"gene,\xff,global_y\n"])
    def test_malformed_header_fails_before_block_discovery(
        self, tmp_path: Path, mocker: MockerFixture, content: bytes
    ) -> None:
        source = _write_source(tmp_path / "transcripts.csv")
        source.path.write_bytes(content)
        parser = mocker.spy(transcript_module, "read_bytes")
        with pytest.raises(ReaderFormatError, match="malformed CSV header"):
            read_transcripts(source, features=_FEATURES, transform=_TRANSFORM)
        parser.assert_not_called()

    def test_bounds_header_read(self, tmp_path: Path, mocker: MockerFixture) -> None:
        source = _write_source(tmp_path / "transcripts.csv")
        mocker.patch.object(transcript_module, "_HEADER_BYTES", 32)
        parser = mocker.spy(transcript_module, "read_bytes")
        with pytest.raises(ReaderFormatError, match="oversized CSV header"):
            read_transcripts(source, features=_FEATURES, transform=_TRANSFORM)
        parser.assert_not_called()

    @pytest.mark.parametrize("record", ["40,GeneB,1,2\n", "40,GeneB,1,2,0,-1,0,1,t,9,8,annotation,excess\n"])
    def test_projection_cannot_hide_bad_record_width(self, tmp_path: Path, mocker: MockerFixture, record: str) -> None:
        source = _write_source(tmp_path / "transcripts.csv")
        with source.path.open("a", encoding="utf-8") as output:
            output.write(record)
        mocker.patch.object(transcript_module, "_BLOCK_BYTES", 256)
        points = read_transcripts(source, features=_FEATURES, transform=_TRANSFORM)
        assert not points.partitions[0].compute().empty
        with pytest.raises(ReaderFormatError, match="malformed transcript CSV"):
            points.partitions[-1].compute()

    def test_model_failure_restores_warning_filters(self, tmp_path: Path, mocker: MockerFixture) -> None:
        source = _write_source(tmp_path / "transcripts.csv")
        parser = mocker.patch.object(transcript_module.PointsModel, "parse", side_effect=RuntimeError("injected"))
        initial = list(warnings.filters)
        with pytest.raises(RuntimeError, match="injected"):
            read_transcripts(source, features=_FEATURES, transform=_TRANSFORM)
        assert warnings.filters == initial
        parser.assert_called_once()
        assert parser.call_args.kwargs["sort"] is False

    def test_never_samples_data_records_at_construction(self, tmp_path: Path, mocker: MockerFixture) -> None:
        source = _write_source(tmp_path / "transcripts.csv")
        source.path.write_text(
            "gene,global_x,global_y,ignored\nGeneB,0.10000000000000002,2," + "a" * (128 * 1024) + "\n",
            encoding="utf-8",
        )
        storage = mocker.spy(transcript_module, "read_bytes")
        parser = mocker.spy(transcript_module, "_read_partition")
        points = read_transcripts(source, features=_FEATURES, transform=_TRANSFORM)
        parser.assert_not_called()
        assert storage.call_args.kwargs["sample"] is False
        result = points.compute()
        assert "ignored" not in result
        assert result["x"].iloc[0] == float("0.10000000000000002")

    def test_literal_glob_characters_in_source_path(self, tmp_path: Path) -> None:
        root = tmp_path / "[region]*?"
        root.mkdir()
        source = _write_source(root / "transcripts.csv")
        result = read_transcripts(source, features=_FEATURES, transform=_TRANSFORM).compute()
        assert len(result) == 40
