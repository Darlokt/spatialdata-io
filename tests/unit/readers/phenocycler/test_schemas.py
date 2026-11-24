"""Column roles and channel identity in marker-list and HuBMAP sidecars."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers.phenocycler._schemas import HUBMAP_CHANNEL_FIELDS, channel_schema, hubmap_annotations

if TYPE_CHECKING:
    from pathlib import Path

_CONTEXT = ReaderErrorContext("phenocycler", "channels.csv")


class TestChannelSchema:
    def test_marker_list_with_unrelated_ids(self, tmp_path: Path) -> None:
        actual = channel_schema(("Timestamp", " marker ", "Channel ID"), context=_CONTEXT, path=tmp_path)
        assert actual == pa.schema([(" marker ", pa.string())])

    @pytest.mark.parametrize("header", [("Name",), ("Marker", "MARKER")])
    def test_absent_or_ambiguous_marker(self, tmp_path: Path, header: tuple[str, ...]) -> None:
        with pytest.raises(ReaderFormatError, match="one named Marker column"):
            channel_schema(header, context=_CONTEXT, path=tmp_path)

    def test_hubmap_identifiers_and_roles_in_arbitrary_column_order(self, tmp_path: Path) -> None:
        header = (*reversed(HUBMAP_CHANNEL_FIELDS.names), "note")
        assert channel_schema(header, context=_CONTEXT, path=tmp_path) == HUBMAP_CHANNEL_FIELDS

    @pytest.mark.parametrize(
        "header", [("channel id",), ("Marker", "is antibody"), (*HUBMAP_CHANNEL_FIELDS.names, "CHANNEL ID")]
    )
    def test_incomplete_or_ambiguous_hubmap_header(self, tmp_path: Path, header: tuple[str, ...]) -> None:
        with pytest.raises(ReaderFormatError, match="HuBMAP"):
            channel_schema(header, context=_CONTEXT, path=tmp_path)


class TestHubmapAnnotations:
    @pytest.mark.parametrize(
        "identifiers", [["one", "one"], ["", "two"], ["/local/channel", "two"], ["one", "t\x00wo"]]
    )
    def test_invalid_identifiers(self, tmp_path: Path, identifiers: list[str]) -> None:
        table = pa.Table.from_arrays(
            [identifiers, [True, False], [False, True], [True, True]], schema=HUBMAP_CHANNEL_FIELDS
        )
        with pytest.raises(ReaderFormatError, match="channel IDs"):
            hubmap_annotations(table, 2, context=_CONTEXT, path=tmp_path)

    def test_missing_role(self, tmp_path: Path) -> None:
        table = pa.Table.from_arrays([["Channel:0:2"], [True], [False], [None]], schema=HUBMAP_CHANNEL_FIELDS)
        with pytest.raises(ReaderFormatError, match="null channel annotations"):
            hubmap_annotations(table, 1, context=_CONTEXT, path=tmp_path)
