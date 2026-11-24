"""Public option validation before visiting the dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from spatialdata_io import phenocycler

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class TestPhenocycler:
    @pytest.mark.parametrize(
        ("value", "error", "message"),
        [
            ("DAPI", TypeError, "sequence"),
            ((1,), TypeError, "contain strings"),
            ((), ValueError, "unique, nonempty"),
            ((" ",), ValueError, "unique, nonempty"),
            (("CD3", "CD3"), ValueError, "unique, nonempty"),
        ],
    )
    def test_invalid_channel_names(self, tmp_path: Path, value: object, error: type[Exception], message: str) -> None:
        with pytest.raises(error, match=message):
            phenocycler(tmp_path / "missing.qptiff", channel_names=cast("Sequence[str]", value))

    def test_bit_depth_option_requires_boolean(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="read_all_bit_depths must be a boolean"):
            phenocycler(tmp_path / "missing.qptiff", read_all_bit_depths=cast("bool", "yes"))
