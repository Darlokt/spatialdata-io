"""Tests for the model-independent reader provenance boundary."""

from __future__ import annotations

from collections import UserDict

from spatialdata_io import __version__
from spatialdata_io.readers._utils.provenance import set_reader_provenance


class TestSetReaderProvenance:
    def test_sets_required_values_and_preserves_unrelated_attributes(self) -> None:
        unrelated = object()
        attrs: dict[str, object] = {"unrelated": unrelated}

        set_reader_provenance(attrs, reader="visium")

        assert attrs == {
            "unrelated": unrelated,
            "spatialdata_io_software_version": __version__,
            "spatialdata_io_reader": "visium",
        }

    def test_sets_format_version_and_replaces_owned_values(self) -> None:
        attrs: UserDict[str, object] = UserDict(
            {
                "spatialdata_io_software_version": "stale-software",
                "spatialdata_io_reader": "stale-reader",
                "spatialdata_io_format_version": "stale-format",
                "unrelated": 1,
            }
        )

        set_reader_provenance(attrs, reader="xenium", format_version="3.0.1")

        assert attrs == {
            "spatialdata_io_software_version": __version__,
            "spatialdata_io_reader": "xenium",
            "spatialdata_io_format_version": "3.0.1",
            "unrelated": 1,
        }

    def test_removes_stale_format_version_when_none(self) -> None:
        attrs: dict[str, object] = {
            "spatialdata_io_format_version": "stale-format",
            "unrelated": "preserved",
        }
        identity = id(attrs)

        set_reader_provenance(attrs, reader="codex", format_version=None)

        assert id(attrs) == identity
        assert attrs == {
            "spatialdata_io_software_version": __version__,
            "spatialdata_io_reader": "codex",
            "unrelated": "preserved",
        }
