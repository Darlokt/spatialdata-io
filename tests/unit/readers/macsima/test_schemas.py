"""MACSiQView source vocabularies, physical units, and the channel-table contract."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from ome_types.model import UnitsLength

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers.macsima._schemas import CHANNEL_VAR_DTYPES, normalize_image_type, pixel_size_um

if TYPE_CHECKING:
    from spatialdata_io.readers.macsima._schemas import SchemaMajor

_CONTEXT = ReaderErrorContext("macsima", "channel image", detected_version="1.1.1")
_PATH = Path("channel.tif")


class TestNormalizeImageType:
    """Each software major has its own acquisition vocabulary."""

    @pytest.mark.parametrize(
        ("value", "major", "expected"),
        [
            ("AntigenCycle", 0, "stain"),
            ("BleachCycle", 0, "bleach"),
            ("S", 1, "stain"),
            ("B", 1, "bleach"),
            ("AF", 1, "autofluorescence"),
        ],
    )
    def test_normalizes_known_values(self, value: str, major: SchemaMajor, expected: str) -> None:
        assert normalize_image_type(value, major=major, context=_CONTEXT, path=_PATH) == expected

    @pytest.mark.parametrize(("value", "major"), [("AF", 0), ("S", 0), ("AntigenCycle", 1), ("stain", 1)])
    def test_rejects_values_of_another_version_or_unknown(self, value: str, major: SchemaMajor) -> None:
        with pytest.raises(ReaderFormatError, match=rf"detected version '1.1.1'.*found image type '{value}'"):
            normalize_image_type(value, major=major, context=_CONTEXT, path=_PATH)


class TestPixelSizeUm:
    """Physical pixel sizes become micrometers with one correctly rounded division."""

    def test_micrometers_are_unchanged(self) -> None:
        assert pixel_size_um(0.17, UnitsLength.MICROMETER, context=_CONTEXT, path=_PATH) == 0.17

    def test_nanometers_are_divided_exactly(self) -> None:
        assert pixel_size_um(170.0, UnitsLength.NANOMETER, context=_CONTEXT, path=_PATH) == 0.17

    @pytest.mark.parametrize("unit", [UnitsLength.MILLIMETER, UnitsLength.PIXEL, UnitsLength.REFERENCEFRAME])
    def test_rejects_other_units(self, unit: UnitsLength) -> None:
        with pytest.raises(ReaderFormatError, match="µm or nm"):
            pixel_size_um(1.0, unit, context=_CONTEXT, path=_PATH)

    @pytest.mark.parametrize("size", [None, 0.0, -1.0, float("inf"), float("nan")])
    def test_rejects_missing_or_nonpositive_sizes(self, size: float | None) -> None:
        with pytest.raises(ReaderFormatError, match="finite positive size"):
            pixel_size_um(size, UnitsLength.MICROMETER, context=_CONTEXT, path=_PATH)


class TestChannelVarDtypes:
    """The channel table keeps the legacy column names in a fixed order."""

    def test_columns_and_types_in_output_order(self) -> None:
        assert list(CHANNEL_VAR_DTYPES.items()) == [
            ("name", "string"),
            ("cycle", "int64"),
            ("imagetype", "string"),
            ("well", "string"),
            ("ROI", "Int64"),
            ("fluorophore", "string"),
            ("clone", "string"),
            ("exposure", "float64"),
        ]
