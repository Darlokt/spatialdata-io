"""Steinbock public-option validation before artifact I/O."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers.steinbock import steinbock

if TYPE_CHECKING:
    from pathlib import Path


class TestSteinbock:
    """Reject malformed public options in signature order before discovering files."""

    @pytest.mark.parametrize(
        ("masks_dir", "error"), [("/masks", ValueError), ("../masks", ValueError), (1, TypeError), (None, TypeError)]
    )
    def test_invalid_masks_dir_precedes_path_io(
        self, tmp_path: Path, masks_dir: object, error: type[Exception]
    ) -> None:
        with pytest.raises(error, match="masks_dir"):
            steinbock(tmp_path / "absent", masks_dir=masks_dir)  # pyrefly: ignore[bad-argument-type] - Exercise an untyped caller.

    def test_invalid_table_file_precedes_path_io(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="table_file"):
            steinbock(tmp_path / "absent", table_file=1)  # pyrefly: ignore[bad-argument-type] - Exercise an untyped caller.

    def test_invalid_chunks_precede_path_io(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="positive"):
            steinbock(tmp_path / "absent", chunks=0)

    def test_invalid_path_policy_precedes_path_io(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="path_policy"):
            steinbock(tmp_path / "absent", path_policy="lenient")  # pyrefly: ignore[bad-argument-type] - Exercise an untyped caller.

    def test_options_are_validated_in_signature_order(self, tmp_path: Path) -> None:
        absent = tmp_path / "absent"
        policy = "lenient"
        with pytest.raises(ValueError, match="masks_dir"):
            steinbock(absent, masks_dir="..", table_file=1, scale_factors=(1,), path_policy=policy)  # pyrefly: ignore[bad-argument-type] - Exercise an untyped caller.
        with pytest.raises(TypeError, match="table_file"):
            steinbock(absent, table_file=1, scale_factors=(1,), path_policy=policy)  # pyrefly: ignore[bad-argument-type] - Exercise an untyped caller.
        with pytest.raises(ValueError, match="scale_factors"):
            steinbock(absent, scale_factors=(1,), path_policy=policy)  # pyrefly: ignore[bad-argument-type] - Exercise an untyped caller.
