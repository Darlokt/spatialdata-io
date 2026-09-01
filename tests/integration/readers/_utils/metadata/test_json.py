"""Integration workflows for bounded JSON metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.metadata import read_json_object
from spatialdata_io.readers._utils.paths import ArtifactFile, normalize_dataset_root, require_file

if TYPE_CHECKING:
    from pathlib import Path


class TestReadJsonObject:
    def test_composes_with_validated_canonical_file_and_returns_resource_independent_data(
        self,
        tmp_path: Path,
    ) -> None:
        context = ReaderErrorContext(reader="xenium", artifact="experiment.xenium", detected_version="3.0")
        root = normalize_dataset_root(tmp_path, context=context)
        target = tmp_path / "metadata-target.json"
        target.write_text('{"analysis":{"version":"3.0"}}', encoding="utf-8")
        alias = tmp_path / "experiment.xenium"
        alias.symlink_to(target)
        artifact = require_file(alias, context=context)
        assert_type(artifact, ArtifactFile)

        result = read_json_object(artifact, max_bytes=128, context=context)
        assert_type(result, dict[str, object])

        assert root == tmp_path.resolve()
        assert artifact == target.resolve()
        target.unlink()
        assert result == {"analysis": {"version": "3.0"}}

    def test_reports_canonical_artifact_and_complete_context_after_path_composition(self, tmp_path: Path) -> None:
        context = ReaderErrorContext(reader="visium_hd", artifact="bin scale factors", detected_version="3")
        target = tmp_path / "scale-target.json"
        target.write_text("[]", encoding="utf-8")
        alias = tmp_path / "scalefactors_json.json"
        alias.symlink_to(target)
        artifact = require_file(alias, context=context)

        with pytest.raises(ReaderFormatError) as raised:
            read_json_object(artifact, max_bytes=2, context=context)

        assert raised.value.context is context
        assert raised.value.path == target.resolve()
        assert "detected version '3'" in str(raised.value)
        assert str(target.resolve()) in str(raised.value)
