"""MCMICRO workflow-mode and marker-panel parsing under bounded resource ownership."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.mcmicro._metadata import read_markers, read_workflow

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _write(tmp_path: Path, text: str, name: str = "params.yml") -> ArtifactFile:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return ArtifactFile(path)


class TestReadWorkflow:
    """Only an actual YAML boolean at workflow.tma selects the layout."""

    @pytest.mark.parametrize(
        ("text", "mode"),
        [
            ("workflow:\n  tma: false\n", "wsi"),
            ("workflow:\n  tma: true\n", "tma"),
            ("workflow:\n  tma: True\n  start-at: registration\nmodules: {}\n", "tma"),
            ("workflow: {tma: no}\n", "wsi"),
            ("﻿workflow:\n  tma: on\n", "tma"),
        ],
    )
    def test_boolean_modes(self, tmp_path: Path, text: str, mode: str) -> None:
        assert read_workflow(_write(tmp_path, text)) == mode

    def test_unrelated_aliases_are_not_expanded(self, tmp_path: Path) -> None:
        text = "workflow:\n  tma: false\nbomb: &a [*a, *a]\n"
        assert read_workflow(_write(tmp_path, text)) == "wsi"

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ('workflow:\n  tma: "false"\n', "workflow.tma tagged"),
            ("workflow:\n  tma: 1\n", "workflow.tma tagged"),
            ("workflow:\n  tma: [true]\n", "workflow.tma tagged"),
            ("workflow:\n  tma: true\n  tma: false\n", "2 'tma' entries"),
            ("workflow:\n  tma: true\nworkflow:\n  tma: true\n", "2 'workflow' entries"),
            ("workflow:\n  other: 1\n", "0 'tma' entries"),
            ("workflow: [tma]\n", "mapping"),
            ("- workflow\n", "mapping"),
            ("", "empty document"),
            ("workflow: [unclosed\n", "malformed YAML"),
            ("workflow:\n  tma: false\n---\nsecond: 1\n", "malformed YAML"),
        ],
    )
    def test_rejects(self, tmp_path: Path, text: str, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            read_workflow(_write(tmp_path, text))

    def test_excessive_depth(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="malformed YAML"):
            read_workflow(_write(tmp_path, "[" * 100_000 + "]" * 100_000))

    def test_size_bound(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch("spatialdata_io.readers.mcmicro._metadata._MAX_PARAMS_BYTES", 8)
        with pytest.raises(ReaderFormatError, match="more than 8 bytes"):
            read_workflow(_write(tmp_path, "workflow:\n  tma: false\n"))

    def test_invalid_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "params.yml"
        path.write_bytes(b"workflow:\n  tma: \xff\n")
        with pytest.raises(ReaderFormatError, match="invalid UTF-8"):
            read_workflow(ArtifactFile(path))


class TestReadMarkers:
    """The panel keeps quoted text exactly and rejects malformed records."""

    def test_quoted_text_and_order(self, tmp_path: Path) -> None:
        panel = read_markers(_write(tmp_path, 'marker_name,note\nDAPI,"a, b"\nCD3,\nDAPI,0001\n', "markers.csv"))
        assert panel.features == ("DAPI_1", "CD3", "DAPI_2")
        assert panel.var["note"].tolist() == ["a, b", "", "0001"]

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("", "unique nonempty"),
            ("marker_name,marker_name\nA,B\n", "unique nonempty"),
            ('"unterminated\n', "malformed CSV header|unique nonempty"),
            ("marker_name\n", "no markers"),
            ("marker_name,x\nA\n", "malformed marker CSV"),
            ("channel_number\n1\n", "marker_name"),
        ],
    )
    def test_rejects(self, tmp_path: Path, text: str, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            read_markers(_write(tmp_path, text, "markers.csv"))
