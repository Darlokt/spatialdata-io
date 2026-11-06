"""Declared OME lengths and channel identities, independently of reader policy."""

from __future__ import annotations

from pathlib import Path

import pytest
from ome_types import OME
from ome_types.model import Channel, Image, Pixels, UnitsLength

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import TiffMetadata, ome_channel_names, ome_pixel_sizes_um
from spatialdata_io.readers._utils.paths import ArtifactFile

_CONTEXT = ReaderErrorContext("test", "OME calibration")
_PATH = Path("source.tif")


def _pixels() -> Pixels:
    return Pixels(dimension_order="XYZCT", type="uint16", size_x=8, size_y=5, size_z=1, size_c=1, size_t=1)


def _metadata(*pixels: Pixels) -> TiffMetadata:
    return TiffMetadata(ArtifactFile(_PATH), (), OME(images=[Image(pixels=value) for value in pixels]))


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

    @pytest.mark.parametrize(
        ("unit", "value", "expected"), [("nm", 9.0, 0.009), ("nm", 170.0, 0.17), ("mm", 0.017, 17.0)]
    )
    def test_conversion_is_correctly_rounded(self, unit: str, value: float, expected: float) -> None:
        pixels = _pixels()
        pixels.physical_size_x = value
        pixels.physical_size_x_unit = UnitsLength(unit)

        assert ome_pixel_sizes_um(_metadata(pixels), series=0, axes=("x",), context=_CONTEXT) == (expected,)
        # Multiplication by the inexact 0.001 gives 0.009000000000000001 for 9 nm.
        assert 9.0 * 0.001 != 0.009

    def test_absent_ome_and_sizes_and_default_unit(self) -> None:
        assert ome_pixel_sizes_um(
            TiffMetadata(ArtifactFile(_PATH), (), None), series=0, axes=("x", "y"), context=_CONTEXT
        ) == (
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


def _channels(
    *names: str | None, ids: tuple[str, ...] | None = None, samples: tuple[int | None, ...] | None = None
) -> TiffMetadata:
    pixels = _pixels()
    identifiers = ids or tuple(f"Channel:0:{index}" for index in range(len(names)))
    sizes = samples or (None,) * len(names)
    pixels.size_c = sum(size or 1 for size in sizes)
    pixels.channels = [
        Channel(id=identifier, name=name, samples_per_pixel=size)
        for identifier, name, size in zip(identifiers, names, sizes, strict=True)
    ]
    return _metadata(pixels)


class TestOmeChannelNames:
    """Prefer unique nonempty names, then unique IDs, expand samples, and never guess positions."""

    def test_unique_names_in_ome_order(self) -> None:
        metadata = _channels("DAPI", "CD3")

        assert ome_channel_names(metadata, series=0, count=2, samples=False, context=_CONTEXT) == ["DAPI", "CD3"]

    @pytest.mark.parametrize("names", [("DAPI", "DAPI"), ("DAPI", None), ("DAPI", "")])
    def test_repeated_or_missing_names_use_ids(self, names: tuple[str | None, ...]) -> None:
        assert ome_channel_names(_channels(*names), series=0, count=2, samples=False, context=_CONTEXT) == [
            "Channel:0:0",
            "Channel:0:1",
        ]

    @pytest.mark.parametrize("samples", [False, True])
    def test_without_ome_returns_none(self, *, samples: bool) -> None:
        metadata = TiffMetadata(ArtifactFile(_PATH), (), None)

        assert ome_channel_names(metadata, series=0, count=3, samples=samples, context=_CONTEXT) is None

    def test_selected_series(self) -> None:
        first, second = _pixels(), _pixels()
        first.channels = [Channel(id="Channel:0:0", name="DAPI")]
        second.channels = [Channel(id="Channel:1:0", name="HE", samples_per_pixel=3)]

        assert ome_channel_names(_metadata(first, second), series=1, count=3, samples=True, context=_CONTEXT) == [
            "HE:sample:0",
            "HE:sample:1",
            "HE:sample:2",
        ]

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            (None, ["Channel:0:0:sample:0", "Channel:0:0:sample:1", "Channel:0:0:sample:2"]),
            ("HE", ["HE:sample:0", "HE:sample:1", "HE:sample:2"]),
        ],
    )
    def test_rgb_channel_expands_to_its_samples(self, name: str | None, expected: list[str]) -> None:
        """OME declares interleaved RGB as one channel record with three samples per pixel."""
        metadata = _channels(name, samples=(3,))

        assert ome_channel_names(metadata, series=0, count=3, samples=True, context=_CONTEXT) == expected

    def test_multiple_channels_expand_in_ome_then_sample_order(self) -> None:
        metadata = _channels("before", "after", samples=(3, 3))

        assert ome_channel_names(metadata, series=0, count=6, samples=True, context=_CONTEXT) == [
            "before:sample:0",
            "before:sample:1",
            "before:sample:2",
            "after:sample:0",
            "after:sample:1",
            "after:sample:2",
        ]

    def test_mixed_and_absent_sample_counts(self) -> None:
        """An absent SamplesPerPixel counts as one sample; counts may differ between records."""
        metadata = _channels("RG", "B", "DAPI", samples=(2, None, 1))

        assert ome_channel_names(metadata, series=0, count=4, samples=True, context=_CONTEXT) == [
            "RG:sample:0",
            "RG:sample:1",
            "B:sample:0",
            "DAPI:sample:0",
        ]

    def test_channel_axis_names_records_regardless_of_samples(self) -> None:
        """Without a samples channel axis, each OME record names one raster channel."""
        metadata = _channels("left", "right", samples=(3, 3))

        assert ome_channel_names(metadata, series=0, count=2, samples=False, context=_CONTEXT) == ["left", "right"]

    def test_samples_fall_back_to_ids_and_stay_unique(self) -> None:
        """Expanded IDs remain unique, even when an ID resembles another expanded identifier."""
        metadata = _channels("same", "same", ids=("Channel:0", "Channel:0:sample:0"), samples=(2, 1))

        assert ome_channel_names(metadata, series=0, count=3, samples=True, context=_CONTEXT) == [
            "Channel:0:sample:0",
            "Channel:0:sample:1",
            "Channel:0:sample:0:sample:0",
        ]

    @pytest.mark.parametrize(
        ("names", "samples", "count", "message"),
        [
            (("DAPI", "CD3"), False, 3, "2 OME channels for 3 raster channels"),
            (("HE",), False, 3, "1 OME channels for 3 raster channels"),
            (("HE",), True, 4, "1 OME channels with 3 samples for 4 raster channels"),
            (("HE", "IF"), True, 3, "2 OME channels with 6 samples for 3 raster channels"),
        ],
    )
    def test_rejects_count_mismatch(self, names: tuple[str, ...], *, samples: bool, count: int, message: str) -> None:
        metadata = _channels(*names, samples=(3,) * len(names))

        with pytest.raises(ReaderFormatError, match=message) as raised:
            ome_channel_names(metadata, series=0, count=count, samples=samples, context=_CONTEXT)

        assert raised.value.context == _CONTEXT
        assert raised.value.path == _PATH

    @pytest.mark.parametrize(("samples", "count"), [(False, 2), (True, 6)])
    def test_rejects_duplicate_ids_when_names_are_not_unique(self, *, samples: bool, count: int) -> None:
        metadata = _channels("DAPI", "DAPI", ids=("Channel:0", "Channel:0"), samples=(3, 3))

        with pytest.raises(ReaderFormatError, match="unique OME channel IDs"):
            ome_channel_names(metadata, series=0, count=count, samples=samples, context=_CONTEXT)
