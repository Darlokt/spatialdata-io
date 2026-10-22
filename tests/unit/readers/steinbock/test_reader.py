"""Steinbock public-option validation before artifact I/O."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers.steinbock import steinbock

if TYPE_CHECKING:
    from pathlib import Path


class TestSteinbock:
    """Reject malformed public configuration before discovering files."""

    def test_invalid_kind_precedes_path_io(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="labels_kind"):
            steinbock(tmp_path / "absent", labels_kind="other")  # type: ignore[bad-argument-type, arg-type]  # Exercise an untyped caller.

    def test_invalid_chunks_precede_path_io(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="positive"):
            steinbock(tmp_path / "absent", chunks=0)
