"""Matrix Market text boundaries used by the 10x count reader."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.counts import MtxMember, _10x_mtx
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.paths import ArtifactFile

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestBlocks:
    def test_arbitrary_block_alignment_preserves_records(self, mocker: MockerFixture) -> None:
        mocker.patch.object(_10x_mtx, "_BLOCK_BYTES", 17)
        source = b"1 2 4\n\t2  3 +6 \r\n  % comment\n3 4 007"
        records = [line for block in _10x_mtx._blocks(io.BytesIO(source)) for line in block.splitlines() if line]
        assert records == [b"1 2 4", b"2 3 6", b"3 4 007"]

    def test_skips_comment_only_block(self, mocker: MockerFixture) -> None:
        mocker.patch.object(_10x_mtx, "_BLOCK_BYTES", 8)
        assert list(_10x_mtx._blocks(io.BytesIO(b"% hello\n1 1 2\n"))) == [b"1 1 2\n"]

    def test_rejects_unbounded_line(self, mocker: MockerFixture) -> None:
        mocker.patch.object(_10x_mtx, "_BLOCK_BYTES", 8)
        with pytest.raises(ValueError, match="record exceeds"):
            tuple(_10x_mtx._blocks(io.BytesIO(b"1 2 " + b"7" * 20)))


class TestDimensions:
    def test_rejects_unbounded_header_comment(self, mocker: MockerFixture) -> None:
        mocker.patch.object(_10x_mtx, "_BLOCK_BYTES", 8)
        with pytest.raises(ValueError, match="size or comment"):
            _10x_mtx._dimensions(io.BytesIO(b"%" * 16 + b"\n1 1 0\n"))


class TestMtxCounts:
    @pytest.mark.parametrize(
        ("dimensions", "records", "reason"),
        [
            ("1 1 2", "1 1 -1\n1 1 2", "negative MatrixMarket count"),
            ("1 1 1", "1 1 1.5", "decimal integer fields"),
            ("1 1 1", "1 1 2e5", "decimal integer fields"),
            ("1 1 1", "1 1 1_0", "decimal integer fields"),
            ("1 1 1", "1 1 7%junk", "decimal integer fields"),
            ("1 1 1", "1 1 0x10", "decimal integer fields"),
            ("1 1 1", "1+1 1 2", "decimal integer fields"),
            ("1 1 2", "1 1 9223372036854775807\n1 1 1", "exceed int64"),
            ("1 1 1", "1 1 9223372036854775808", "invalid value"),
            ("1 1 1", "0 1 7", "outside the declared"),
            ("1 1 1", "1 2 7", "outside the declared"),
            ("1 1 1", "1 1 7 0", "Expected 3 columns"),
            ("1 1 2", "1 1 7", "expected 2 Matrix Market records"),
            ("1 1 0", "1 1 0", "declares 0 records"),
            ("1 1 -1", "", "three nonnegative integer"),
            ("", "", "three nonnegative integer"),
            ("2 1 0", "", "do not match annotations"),
        ],
    )
    def test_rejects_malformed_records(self, tmp_path: Path, dimensions: str, records: str, reason: str) -> None:
        path = ArtifactFile(tmp_path / "matrix.mtx")
        path.write_text(f"%%MatrixMarket matrix coordinate integer general\n{dimensions}\n{records}\n")
        with pytest.raises(ValueError, match=reason):
            _10x_mtx._mtx_counts(
                MtxMember(path, compressed=False), (1, 1), context=ReaderErrorContext("test", "counts")
            )
