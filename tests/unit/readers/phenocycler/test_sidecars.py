"""Bounded, projected and contextual parsing for QPI sidecar files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import require_file
from spatialdata_io.readers.phenocycler._layout import RegionFiles, ScanFiles
from spatialdata_io.readers.phenocycler._sidecars import read_sidecars

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

_CONTEXT = ReaderErrorContext("phenocycler", "QPI scan")


def _files(source: Path) -> RegionFiles:
    path = require_file(source, context=_CONTEXT)
    return RegionFiles(
        ScanFiles("sample", path, None, "sample.qptiff"),
        path if path.suffix == ".xpd" else None,
        path if path.suffix == ".csv" else None,
    )


class TestReadSidecars:
    @pytest.mark.parametrize(
        ("suffix", "data", "failure"),
        [
            (".xpd", b"\x80", "invalid UTF-8"),
            (".xpd", b"{", "malformed JSON"),
            (".xpd", b'{"wells": 42}', "non-array wells"),
            (".csv", b"x\nCD3\n", "one named Marker column"),
            (".csv", b"Marker,Marker\nCD3,CD4\n", "repeated column names"),
            (".csv", b"Marker,Other\nCD3,too,many\n", "CSV"),
        ],
    )
    def test_invalid_metadata_reports_path_and_version(
        self, tmp_path: Path, suffix: str, data: bytes, failure: str
    ) -> None:
        path = tmp_path / f"sample{suffix}"
        path.write_bytes(data)
        with pytest.raises(ReaderFormatError, match=failure) as exc:
            read_sidecars(_files(path), ("CD3",), detected_version="5")
        assert exc.value.context.detected_version == "5"
        assert exc.value.path == path

    @pytest.mark.parametrize("suffix", [".csv", ".xpd"])
    def test_sidecar_size_is_bounded(self, tmp_path: Path, mocker: MockerFixture, suffix: str) -> None:
        mocker.patch("spatialdata_io.readers.phenocycler._sidecars._SIDECAR_BYTES", 7)
        path = tmp_path / f"sample{suffix}"
        path.write_bytes(b"Marker\nCD3\n")
        with pytest.raises(ReaderFormatError, match="exceeding 7 bytes"):
            read_sidecars(_files(path), ("CD3",), detected_version=None)

    def test_reports_mismatched_distinct_marker_counts(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.channels.csv"
        path.write_text("Marker,Ignored\nDAPI,c\nDAPI,d\nCD4,e\nBLANK,foo\n,F\n")
        with pytest.warns(UserWarning, match="2 distinct markers.*1 match"):
            result = read_sidecars(_files(path), ("DAPI", "CD3"), detected_version=None)
        assert result == {"channels_csv": {"marker_count": 2, "matching_marker_count": 1}}

    @pytest.mark.parametrize(
        ("row", "failure"), [("Channel:0:0,Yes,No,other", "CSV"), ("Channel:0:0,Yes,No,No", "one row for each")]
    )
    def test_hubmap_requires_one_complete_row_per_channel(self, tmp_path: Path, row: str, failure: str) -> None:
        path = tmp_path / "sample.channels.csv"
        path.write_text(
            "channel id,is channel used for nuclei segmentation,is channel used for cell segmentation,is antibody\n"
            + row
            + "\n"
        )
        with pytest.raises(ReaderFormatError, match=failure):
            read_sidecars(_files(path), ("DAPI", "CD3"), detected_version="5")
