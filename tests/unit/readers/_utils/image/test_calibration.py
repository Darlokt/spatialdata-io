"""Declared OME length conversion independently of reader registration policy."""

from __future__ import annotations

from pathlib import Path

import pytest
from ome_types import OME
from ome_types.model import Image, Pixels, UnitsLength

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import TiffMetadata
from spatialdata_io.readers._utils.image._calibration import ome_pixel_sizes_um

_CONTEXT = ReaderErrorContext("test", "OME calibration")
_PATH = Path("source.tif")


def _pixels() -> Pixels:
    return Pixels(dimension_order="XYZCT", type="uint16", size_x=8, size_y=5, size_z=1, size_c=1, size_t=1)


def _metadata(*pixels: Pixels) -> TiffMetadata:
    return TiffMetadata(_PATH, (), OME(images=[Image(pixels=value) for value in pixels]))


class TestOmePixelSizesUm:
    """Keep absence distinct from invalid declared metadata, without mutation."""

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    @pytest.mark.parametrize(("unit", "value", "expected"), [("µm", 0.3, 0.3), ("nm", 300.0, 0.3), ("mm", 0.03, 30.0)])
    def test_axis_order_units_and_partial_declarations(
        self, axis: str, unit: str, value: float, expected: float
    ) -> None:
        pixels = _pixels()
        setattr(pixels, f"physical_size_{axis}", value)
        setattr(pixels, f"physical_size_{axis}_unit", unit)
        metadata = _metadata(pixels)
        before = pixels.model_dump()

        actual = ome_pixel_sizes_um(metadata, series=0, axes=("z", "y", "x"), context=_CONTEXT)

        assert actual == tuple(expected if other == axis else None for other in "zyx")
        assert pixels.model_dump() == before

    def test_absent_ome_and_sizes_and_default_unit(self) -> None:
        assert ome_pixel_sizes_um(TiffMetadata(_PATH, (), None), series=0, axes=("x", "y"), context=_CONTEXT) == (
            None,
            None,
        )
        pixels = _pixels()
        metadata = _metadata(pixels)
        assert ome_pixel_sizes_um(metadata, series=0, axes=("x", "y"), context=_CONTEXT) == (None, None)
        pixels.physical_size_x = 2
        pixels.physical_size_y = 3
        assert ome_pixel_sizes_um(metadata, series=0, axes=("x", "y"), context=_CONTEXT) == (2, 3)

    def test_selected_series(self) -> None:
        first, second = _pixels(), _pixels()
        first.physical_size_x = 1
        second.physical_size_x = 2
        assert ome_pixel_sizes_um(_metadata(first, second), series=1, axes=("x",), context=_CONTEXT) == (2,)

    @pytest.mark.parametrize(
        ("value", "unit"),
        [
            (0.0, "µm"),
            (-1.0, "µm"),
            (float("inf"), "µm"),
            (float("nan"), "µm"),
            (1e308, "mm"),
            (5e-324, "nm"),
            (1.0, "pixel"),
        ],
    )
    def test_malformed_even_with_missing_other_axis(self, value: float, unit: str) -> None:
        pixels = _pixels()
        # Bypass Pydantic's own nonpositive-value rejection to exercise conversion
        # invariants independently of the external model's validation policy.
        pixels.__dict__["physical_size_x"] = value
        pixels.physical_size_x_unit = UnitsLength(unit)
        with pytest.raises(ReaderFormatError, match="PhysicalSizeX") as caught:
            ome_pixel_sizes_um(_metadata(pixels), series=0, axes=("x", "y"), context=_CONTEXT)
        assert "source.tif" in str(caught.value)
