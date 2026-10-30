"""Scientific X/Y calibration, units and malformed seqFISH metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ome_types.model import OME, Image, Pixels, Pixels_DimensionOrder, PixelType, UnitsLength

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image import TiffMetadata
from spatialdata_io.readers.seqfish._transforms import pixel_scale

if TYPE_CHECKING:
    from pathlib import Path


def _metadata(path: Path) -> TiffMetadata:
    """Produce small owned OME metadata without requiring raster pixels."""
    pixels = Pixels(
        dimension_order=Pixels_DimensionOrder.XYCZT,
        type=PixelType.UINT16,
        size_x=10,
        size_y=20,
        size_c=1,
        size_z=1,
        size_t=1,
        physical_size_x=0.2,
        physical_size_y=0.5,
    )
    return TiffMetadata(path, (), OME(images=[Image(pixels=pixels)]))


class TestPixelScale:
    """Keep each axis and its physical unit tied to the correct coordinate."""

    @pytest.mark.parametrize(
        ("unit", "x", "y"),
        [
            (UnitsLength.MICROMETER, 0.2, 0.5),
            (UnitsLength.NANOMETER, 200.0, 500.0),
            (UnitsLength.MILLIMETER, 0.0002, 0.0005),
        ],
    )
    def test_anisotropic_units_normalize_to_micrometers(
        self, tmp_path: Path, unit: UnitsLength, x: float, y: float
    ) -> None:
        metadata = _metadata(tmp_path / "DAPI.tiff")
        assert metadata.ome is not None
        pixels = metadata.ome.images[0].pixels
        pixels.physical_size_x = x
        pixels.physical_size_y = y
        pixels.physical_size_x_unit = unit
        pixels.physical_size_y_unit = unit

        scale = pixel_scale(metadata)

        assert scale.axes == ("y", "x")
        np.testing.assert_allclose(scale.to_scale_vector(("y", "x")), [0.5, 0.2], rtol=1e-15)
        np.testing.assert_allclose(
            scale.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), np.diag([0.2, 0.5, 1])
        )

    def test_axis_units_can_differ(self, tmp_path: Path) -> None:
        metadata = _metadata(tmp_path / "DAPI.tiff")
        assert metadata.ome is not None
        metadata.ome.images[0].pixels.physical_size_x = 200.0
        metadata.ome.images[0].pixels.physical_size_x_unit = UnitsLength.NANOMETER
        np.testing.assert_array_equal(pixel_scale(metadata).to_scale_vector(("y", "x")), [0.5, 0.2])

    @pytest.mark.parametrize("missing", ["ome", "images", "multiple"])
    def test_missing_or_ambiguous_image_metadata(self, tmp_path: Path, missing: str) -> None:
        metadata = _metadata(tmp_path / "DAPI.tiff")
        ome = metadata.ome
        if missing == "ome":
            metadata = TiffMetadata(metadata.path, (), None)
        else:
            assert ome is not None
            ome.images = [] if missing == "images" else [ome.images[0], ome.images[0]]
        with pytest.raises(ReaderFormatError, match="one OME image"):
            pixel_scale(metadata)

    @pytest.mark.parametrize("axis", ["x", "y"])
    @pytest.mark.parametrize("value", [None, 0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_size(self, tmp_path: Path, axis: str, value: float | None) -> None:
        metadata = _metadata(tmp_path / "DAPI.tiff")
        assert metadata.ome is not None
        # An owned OME object remains externally mutable; test malformed values
        # without asking the OME constructor to reject them before our boundary.
        pixels = metadata.ome.images[0].pixels.model_copy(update={f"physical_size_{axis}": value})
        metadata.ome.images[0].pixels = pixels
        with pytest.raises(ReaderFormatError, match="finite positive physical"):
            pixel_scale(metadata)

    @pytest.mark.parametrize("unit", [UnitsLength.PIXEL, UnitsLength.METER, UnitsLength.REFERENCEFRAME])
    def test_unsupported_units(self, tmp_path: Path, unit: UnitsLength) -> None:
        metadata = _metadata(tmp_path / "DAPI.tiff")
        assert metadata.ome is not None
        metadata.ome.images[0].pixels.physical_size_x_unit = unit
        with pytest.raises(ReaderFormatError, match="µm, nm or mm"):
            pixel_scale(metadata)

    @pytest.mark.parametrize(("value", "unit"), [(1e308, UnitsLength.MILLIMETER), (5e-324, UnitsLength.NANOMETER)])
    def test_conversion_must_remain_finite_and_positive(self, tmp_path: Path, value: float, unit: UnitsLength) -> None:
        metadata = _metadata(tmp_path / "DAPI.tiff")
        assert metadata.ome is not None
        metadata.ome.images[0].pixels.physical_size_x = value
        metadata.ome.images[0].pixels.physical_size_x_unit = unit
        with pytest.raises(ReaderFormatError, match="representable in µm"):
            pixel_scale(metadata)
