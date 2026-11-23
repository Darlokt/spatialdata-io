"""Parquet source lifetimes and format error context."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.xenium._parquet import open_parquet

if TYPE_CHECKING:
    from pathlib import Path

_CONTEXT = ReaderErrorContext("xenium", "transcripts", "4.0.0")


class TestOpenParquet:
    def test_closes_on_success(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "transcripts.parquet")
        pq.write_table(pa.table({"feature_name": ["GeneA"]}), path)
        with open_parquet(path, context=_CONTEXT) as source:
            assert source.read().column("feature_name").to_pylist() == ["GeneA"]
        assert source.closed

    def test_closes_and_contextualizes_consumer_io_failure(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "transcripts.parquet")
        pq.write_table(pa.table({"feature_name": ["GeneA"]}), path)
        message = "interrupted"
        with pytest.raises(OSError, match="interrupted") as caught, open_parquet(path, context=_CONTEXT) as source:
            raise OSError(message)
        assert source.closed
        assert str(path) in caught.value.__notes__[0]
        assert "transcripts" in caught.value.__notes__[0]
