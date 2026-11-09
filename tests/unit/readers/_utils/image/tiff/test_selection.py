"""Tests for caller TIFF index validation and the single-series rule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import TiffMetadata, select_tiff_series, validate_tiff_index
from spatialdata_io.readers._utils.image.tiff._types import TiffLevelMetadata, TiffSeriesMetadata
from spatialdata_io.readers._utils.paths import ArtifactFile

_CONTEXT = ReaderErrorContext("test", "image", "1.0")
_PATH = Path("image.tif")


def _metadata(series: int) -> TiffMetadata:
    level = TiffLevelMetadata((4, 4), np.dtype(np.uint8), ("Y", "X"), (4, 4), None)
    return TiffMetadata(ArtifactFile(_PATH), (TiffSeriesMetadata((level,)),) * series, None)


class TestValidateTiffIndex:
    """Accept nonnegative integers, and None only for an optional selection."""

    @pytest.mark.parametrize("value", [0, 3])
    @pytest.mark.parametrize("optional", [True, False])
    def test_accepts_nonnegative_integers(self, value: int, *, optional: bool) -> None:
        validate_tiff_index(value, parameter="tiff_level", optional=optional)

    def test_accepts_none_when_optional(self) -> None:
        validate_tiff_index(None, parameter="tiff_series", optional=True)

    @pytest.mark.parametrize(
        ("value", "optional", "message"),
        [
            (True, True, "tiff_series must be an integer or None; found bool."),
            (1.0, True, "tiff_series must be an integer or None; found float."),
            ("0", True, "tiff_series must be an integer or None; found str."),
            (np.int64(0), True, "tiff_series must be an integer or None; found int64."),
            (None, False, "tiff_series must be an integer; found NoneType."),
            (False, False, "tiff_series must be an integer; found bool."),
        ],
    )
    def test_rejects_non_integers_naming_the_type(self, value: object, *, optional: bool, message: str) -> None:
        with pytest.raises(TypeError, match=f"^{message}$"):
            validate_tiff_index(value, parameter="tiff_series", optional=optional)

    @pytest.mark.parametrize("optional", [True, False])
    def test_rejects_negative_indexes(self, *, optional: bool) -> None:
        with pytest.raises(ValueError, match=r"^tiff_level must be nonnegative; found -1\.$"):
            validate_tiff_index(-1, parameter="tiff_level", optional=optional)


class TestSelectTiffSeries:
    """Read the only series or an explicit one; never pick the first of several."""

    @pytest.mark.parametrize("option", [None, "tiff_series"])
    def test_only_series(self, option: str | None) -> None:
        assert select_tiff_series(_metadata(1), None, context=_CONTEXT, option=option) == 0

    @pytest.mark.parametrize(("series", "count"), [(0, 1), (1, 3), (5, 2)])
    def test_explicit_selection_is_returned_for_read_tiff_to_range_check(self, series: int, count: int) -> None:
        assert select_tiff_series(_metadata(count), series, context=_CONTEXT, option="tiff_series") == series

    @pytest.mark.parametrize(
        ("option", "expected"),
        [(None, "one TIFF series"), ("tiff_series", "one TIFF series or an explicit tiff_series selection")],
    )
    @pytest.mark.parametrize("count", [0, 2])
    def test_rejects_ambiguous_files_without_a_selection(self, option: str | None, expected: str, count: int) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            select_tiff_series(_metadata(count), None, context=_CONTEXT, option=option)

        assert caught.value.context == _CONTEXT
        assert caught.value.path == _PATH
        assert caught.value.found == f"{count} TIFF series"
        assert caught.value.expected == expected
