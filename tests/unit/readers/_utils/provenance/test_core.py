"""Tests for the model-independent reader provenance boundary."""

from __future__ import annotations

import ast
import inspect
from collections import UserDict
from collections.abc import MutableMapping
from pathlib import Path
from typing import get_type_hints

from spatialdata_io import __version__
from spatialdata_io.readers._utils.provenance import set_reader_provenance

_SOURCE = Path(inspect.getfile(set_reader_provenance))
_ALLOWED_IMPORTS = {"__future__", "collections.abc", "spatialdata_io"}
_ALLOWED_ATTRIBUTE_CALLS = {"pop"}


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

    def test_signature_is_narrow_and_mapping_based(self) -> None:
        signature = inspect.signature(set_reader_provenance)
        parameters = signature.parameters
        annotations = get_type_hints(set_reader_provenance)

        assert set(parameters) == {"attrs", "reader", "format_version"}
        assert parameters["attrs"].kind is inspect.Parameter.POSITIONAL_ONLY
        assert parameters["reader"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["reader"].default is inspect.Parameter.empty
        assert parameters["format_version"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["format_version"].default is None
        assert annotations == {
            "attrs": MutableMapping[str, object],
            "reader": str,
            "format_version": str | None,
            "return": type(None),
        }

    def test_source_has_no_model_artifact_or_data_dependencies(self) -> None:
        tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
        imported_modules = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported_modules.update(
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        )
        direct_calls = {
            node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        attribute_calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

        assert imported_modules <= _ALLOWED_IMPORTS
        assert not direct_calls
        assert attribute_calls <= _ALLOWED_ATTRIBUTE_CALLS
