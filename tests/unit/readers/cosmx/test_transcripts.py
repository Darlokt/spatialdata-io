"""Lazy CosMx transcript partitions for plain and member-aligned gzip sources."""

from __future__ import annotations

import gzip
import io
import pickle
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from dask.callbacks import Callback
from spatialdata.transformations import BaseTransformation, Identity, get_transformation

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.cosmx import _transcripts
from spatialdata_io.readers.cosmx._transcripts import read_transcripts
from spatialdata_io.readers.cosmx._transforms import Placement

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

_HEADER = "fov,cell_ID,x_global_px,y_global_px,x_local_px,y_local_px,z,target,CellComp,extra\n"
_ROWS = [
    f"1,{i % 3},{1000.5 + i},{5000.25 + i},{0.5 + i},{0.25 + i},{i % 4 - 1},{'AB'[i % 2]},Nuclear,x\n"
    for i in range(10)
]
_FEATURES = ("A", "B", "C")
_PLACEMENTS = {1: Placement(1, 1, 1000.0, 5000.0)}


def _plain(tmp_path: Path, rows: list[str] = _ROWS) -> ArtifactFile:
    path = tmp_path / "tx.csv"
    path.write_text(_HEADER + "".join(rows), encoding="utf-8")
    return ArtifactFile(path)


def _members(tmp_path: Path, rows: list[str] = _ROWS, size: int = 3) -> ArtifactFile:
    """Write fwrite-like output: a header member, then independent record-aligned members."""
    path = tmp_path / "tx.csv.gz"
    with path.open("wb") as handle:
        handle.write(gzip.compress(_HEADER.encode(), mtime=0))
        for start in range(0, len(rows), size):
            handle.write(gzip.compress("".join(rows[start : start + size]).encode(), mtime=0))
    return ArtifactFile(path)


def _expected() -> pd.DataFrame:
    return pd.read_csv(io.StringIO(_HEADER + "".join(_ROWS)))


class TestReadTranscripts:
    """Points are lazy, global, typed, and identical for every partitioning."""

    @pytest.mark.parametrize("source", ["plain", "gzip"])
    def test_partitions_reproduce_source_rows(self, tmp_path: Path, mocker: MockerFixture, source: str) -> None:
        mocker.patch("spatialdata_io.readers.cosmx._transcripts._BLOCK_BYTES", 64)
        mocker.patch("spatialdata_io.readers.cosmx._transcripts._GZIP_RANGE_BYTES", 40)
        mocker.patch("spatialdata_io.readers.cosmx._transcripts._READ_BYTES", 16)
        path = _plain(tmp_path) if source == "plain" else _members(tmp_path)
        tasks: list[object] = []
        with Callback(posttask=lambda *args: tasks.append(args)):
            points = read_transcripts(path, features=_FEATURES, placements=_PLACEMENTS)
        assert not tasks
        assert points.npartitions > 1
        result = points.compute()
        expected = _expected()
        np.testing.assert_array_equal(result["x"].to_numpy(), expected["x_global_px"].to_numpy())
        np.testing.assert_array_equal(result["y"].to_numpy(), expected["y_global_px"].to_numpy())
        np.testing.assert_array_equal(result["x_local"].to_numpy(), expected["x_local_px"].to_numpy())
        assert result["target"].tolist() == expected["target"].tolist()
        assert result["target"].cat.categories.tolist() == list(_FEATURES)
        assert result["cell_ID"].tolist() == expected["cell_ID"].tolist()
        assert result["z_raw"].tolist() == expected["z"].tolist()
        assert result.columns.tolist() == [
            "x",
            "y",
            "target",
            "fov",
            "cell_ID",
            "x_local",
            "y_local",
            "z_raw",
            "CellComp",
        ]

    def test_frames_and_serialization(self, tmp_path: Path) -> None:
        points = read_transcripts(_plain(tmp_path), features=_FEATURES, placements=_PLACEMENTS)
        assert isinstance(get_transformation(points, "global"), Identity)
        transform = get_transformation(points, "1")
        assert isinstance(transform, BaseTransformation)
        local = transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        np.testing.assert_array_equal(local, [[1, 0, -1000], [0, 1, -5000], [0, 0, 1]])
        restored = pickle.loads(pickle.dumps(points))  # noqa: S301, RUF100 - Only deserialize this test's own graph.
        assert len(restored.compute()) == len(_ROWS)

    def test_unknown_target_fails_when_its_partition_is_computed(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch("spatialdata_io.readers.cosmx._transcripts._BLOCK_BYTES", 64)
        rows = [*_ROWS[:-1], _ROWS[-1].replace(",B,", ",Z,")]
        points = read_transcripts(_plain(tmp_path, rows), features=_FEATURES, placements=_PLACEMENTS)
        assert len(points.partitions[: points.npartitions - 1].compute()) > 0
        with pytest.raises(ReaderFormatError, match="unknown targets"):
            points.compute()

    @pytest.mark.parametrize(
        ("row", "match"),
        [
            ("1,1,inf,1,1,1,0,A,N,x\n", "nonfinite"),
            ("1,1,1,1,1,1,0,A,N\n", "malformed transcript"),
            ("0,1,1,1,1,1,0,A,N,x\n", "below 1"),
        ],
    )
    def test_invalid_rows_fail_lazily(self, tmp_path: Path, row: str, match: str) -> None:
        points = read_transcripts(_plain(tmp_path, [row]), features=_FEATURES, placements=_PLACEMENTS)
        with pytest.raises(ReaderFormatError, match=match):
            points.compute()

    def test_missing_columns_fail_at_construction(self, tmp_path: Path) -> None:
        path = tmp_path / "tx.csv"
        path.write_text("fov,target\n1,A\n", encoding="utf-8")
        with pytest.raises(ReaderFormatError, match="missing"):
            read_transcripts(ArtifactFile(path), features=_FEATURES, placements=_PLACEMENTS)


class TestGzipLayout:
    """Contiguous record-aligned gzip members are verified without scanning for magic bytes."""

    def test_corrupt_first_owned_member_is_not_skipped(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _members(tmp_path, size=1)
        header = gzip.compress(_HEADER.encode(), mtime=0)
        data = bytearray(path.read_bytes())
        # The second task owns this member; a later intact member must not hide its corrupt CRC.
        data[len(header) + len(gzip.compress(_ROWS[0].encode(), mtime=0)) - 8] ^= 0xFF
        path.write_bytes(data)
        mocker.patch("spatialdata_io.readers.cosmx._transcripts._GZIP_RANGE_BYTES", len(header))
        points = read_transcripts(path, features=_FEATURES, placements=_PLACEMENTS)
        with pytest.raises(ReaderFormatError, match="corrupt gzip member"):
            points.compute()

    def test_header_and_many_rows_in_first_member_stay_lazy(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "tx.csv.gz"
        path.write_bytes(
            gzip.compress((_HEADER + _ROWS[0] * 100_000).encode(), mtime=0) + gzip.compress(_ROWS[1].encode(), mtime=0)
        )
        decoder = mocker.spy(_transcripts, "_decode_member")
        # The decoder owns gzip's read-ahead; this spy measures eager source I/O.
        header_reader = mocker.spy(gzip._GzipReader, "read")  # noqa: SLF001, RUF100
        points = read_transcripts(ArtifactFile(path), features=_FEATURES, placements=_PLACEMENTS)
        decoder.assert_not_called()
        assert 0 < sum(len(block) for block in header_reader.spy_return_list) < 65536
        # Every first-member row is returned once, including when it crosses compressed task ranges.
        assert len(points.compute()) == 100_001

    def test_without_optional_annotations(self, tmp_path: Path) -> None:
        path = tmp_path / "tx.csv"
        path.write_text(
            "fov,cell_ID,x_global_px,y_global_px,x_local_px,y_local_px,target\n1,0,1,2,3,4,A\n", encoding="utf-8"
        )
        result = read_transcripts(ArtifactFile(path), features=_FEATURES, placements=_PLACEMENTS).compute()
        assert result.columns.tolist() == ["x", "y", "target", "fov", "cell_ID", "x_local", "y_local"]

    def test_truncated_final_member_fails(self, tmp_path: Path) -> None:
        path = _members(tmp_path)
        path.write_bytes(path.read_bytes()[:-5])
        points = read_transcripts(path, features=_FEATURES, placements=_PLACEMENTS)
        with pytest.raises(ReaderFormatError, match="corrupt gzip member"):
            points.compute()

    def test_corrupt_header_member_fails(self, tmp_path: Path) -> None:
        path = _members(tmp_path)
        data = bytearray(path.read_bytes())
        header_member = len(gzip.compress(_HEADER.encode(), mtime=0))
        data[header_member - 8] ^= 0xFF
        path.write_bytes(bytes(data))
        with pytest.raises(ReaderFormatError, match=r"corrupt gzip member|invalid gzip data"):
            read_transcripts(path, features=_FEATURES, placements=_PLACEMENTS).compute()

    def test_single_stream_gzip_is_supported(self, tmp_path: Path) -> None:
        path = tmp_path / "tx.csv.gz"
        path.write_bytes(gzip.compress((_HEADER + "".join(_ROWS)).encode(), mtime=0))
        result = read_transcripts(ArtifactFile(path), features=_FEATURES, placements=_PLACEMENTS).compute()
        assert len(result) == len(_ROWS)

    def test_not_gzip_fails_at_construction(self, tmp_path: Path) -> None:
        path = tmp_path / "tx.csv.gz"
        path.write_bytes(b"fov,target\n")
        with pytest.raises(ReaderFormatError, match="invalid gzip data"):
            read_transcripts(ArtifactFile(path), features=_FEATURES, placements=_PLACEMENTS)

    def test_member_ending_inside_a_record_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "tx.csv.gz"
        data = "".join(_ROWS)
        with path.open("wb") as handle:
            handle.write(gzip.compress(_HEADER.encode(), mtime=0))
            handle.write(gzip.compress(data[:30].encode(), mtime=0))
            handle.write(gzip.compress(data[30:].encode(), mtime=0))
        points = read_transcripts(ArtifactFile(path), features=_FEATURES, placements=_PLACEMENTS)
        with pytest.raises(ReaderFormatError, match="record-aligned"):
            points.compute()

    def test_corrupt_member_fails(self, tmp_path: Path) -> None:
        path = _members(tmp_path)
        data = bytearray(path.read_bytes())
        data[-12] ^= 0xFF
        path.write_bytes(bytes(data))
        points = read_transcripts(path, features=_FEATURES, placements=_PLACEMENTS)
        with pytest.raises(ReaderFormatError, match="corrupt gzip member"):
            points.compute()

    def test_crossed_ranges_serialize_and_return_every_row_once(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _members(tmp_path, size=10)
        mocker.patch("spatialdata_io.readers.cosmx._transcripts._GZIP_RANGE_BYTES", 8)
        points = read_transcripts(path, features=_FEATURES, placements=_PLACEMENTS)
        restored = pickle.loads(pickle.dumps(points))  # noqa: S301, RUF100 - Deserialize this test's own graph.
        assert restored.npartitions > 10
        assert len(restored.compute(scheduler="processes")) == len(_ROWS)

    @pytest.mark.parametrize("trailer", [b"junk", b"\x1f\x8b\x08\x00"])
    def test_trailing_corruption_is_not_skipped(self, tmp_path: Path, trailer: bytes) -> None:
        path = _members(tmp_path)
        path.write_bytes(path.read_bytes() + trailer)
        points = read_transcripts(path, features=_FEATURES, placements=_PLACEMENTS)
        with pytest.raises(ReaderFormatError, match="corrupt gzip member"):
            points.compute()

    @pytest.mark.parametrize("corrupt", [False, True])
    def test_lazy_sources_close_on_success_and_failure(
        self, tmp_path: Path, mocker: MockerFixture, *, corrupt: bool
    ) -> None:
        path = _members(tmp_path)
        if corrupt:
            path.write_bytes(path.read_bytes()[:-4])
        points = read_transcripts(path, features=_FEATURES, placements=_PLACEMENTS)
        sources = mocker.spy(type(path), "open")
        if corrupt:
            with pytest.raises(ReaderFormatError, match="corrupt gzip member"):
                points.compute(scheduler="single-threaded")
        else:
            assert len(points.compute(scheduler="single-threaded")) == len(_ROWS)
        assert sources.spy_return_list
        assert all(handle.closed for handle in sources.spy_return_list)
