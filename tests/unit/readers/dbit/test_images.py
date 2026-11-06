"""Test DBiT image options, axes, channel interpretation, and pyramids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import imagecodecs
import numpy as np
import pytest
import tifffile
from xarray import DataArray, DataTree

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import inspect_tiff, make_raster_options
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.dbit._images import read_image, validate_image_options

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

DEFAULT_OPTIONS = make_raster_options()


_RASTER_CONTEXT = ReaderErrorContext("test", "raster")


class TestValidateImageOptions:
    """Reject malformed caller options before loading metadata or expression."""

    @pytest.mark.parametrize("axes", ["", "YYX", "CYXS", "C YX", "YX1", "CY", "\u03b1yx"])
    def test_invalid_axes(self, axes: str) -> None:
        with pytest.raises(ValueError, match="data_axes"):
            validate_image_options(axes, None, 0)

    @pytest.mark.parametrize("axes", [["y", "x"], ("c", "y", "x"), b"YX"])
    def test_axes_must_be_a_string(self, axes: object) -> None:
        with pytest.raises(TypeError, match="data_axes must be a string"):
            validate_image_options(axes, None, 0)  # pyrefly: ignore[bad-argument-type] - Untrusted caller input.

    @pytest.mark.parametrize("value", [True, 0.5, "0"])
    @pytest.mark.parametrize("option", ["series", "level"])
    def test_tiff_selection_type(self, value: object, option: str) -> None:
        series, level = (value, 0) if option == "series" else (None, value)
        with pytest.raises(TypeError, match=f"tiff_{option} must be an integer"):
            validate_image_options(None, series, level)  # pyrefly: ignore[bad-argument-type] - Untrusted caller input.

    def test_tiff_level_cannot_be_none(self) -> None:
        with pytest.raises(TypeError, match="tiff_level must be an integer"):
            validate_image_options(None, None, None)  # pyrefly: ignore[bad-argument-type] - Untrusted caller input.

    @pytest.mark.parametrize("option", ["series", "level"])
    def test_negative_tiff_selection(self, option: str) -> None:
        series, level = (-1, 0) if option == "series" else (None, -1)
        with pytest.raises(ValueError, match=f"tiff_{option} must be nonnegative; found -1"):
            validate_image_options(None, series, level)

    @pytest.mark.parametrize("axes", [None, "YX", "yxs", "YXS", "TZCYX"])
    @pytest.mark.parametrize("series", [None, 0, 1])
    def test_valid_options(self, axes: str | None, series: int | None) -> None:
        validate_image_options(axes, series, 2)


class TestReadImage:
    """Preserve scientific channels and named singleton dimensions lazily."""

    @pytest.mark.parametrize("shape", [(1, 1), (1, 6), (7, 1)])
    def test_grayscale_preserves_spatial_singletons(self, tmp_path: Path, shape: tuple[int, int]) -> None:
        path = tmp_path / "image.tif"
        source = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        tifffile.imwrite(path, source, metadata={"axes": "YX"})

        image = read_image(ArtifactFile(path), data_axes=None, tiff_series=None, tiff_level=0, raster=DEFAULT_OPTIONS)

        assert isinstance(image, DataArray)
        assert image.dims == ("c", "y", "x")
        assert image.shape == (1, *shape)
        np.testing.assert_array_equal(image.data.compute(), source[None])

    @pytest.mark.parametrize("channels", [1, 3, 4, 5])
    def test_scientific_channels_keep_positional_labels(self, tmp_path: Path, channels: int) -> None:
        path = tmp_path / "image.tif"
        source = np.arange(channels * 30, dtype=np.uint16).reshape(channels, 5, 6)
        tifffile.imwrite(path, source, photometric="minisblack", metadata={"axes": "CYX"})

        image = read_image(
            ArtifactFile(path), data_axes=None, tiff_series=None, tiff_level=0, raster=make_raster_options(chunks=2)
        )

        assert isinstance(image, DataArray)
        assert image.coords["c"].to_numpy().tolist() == list(range(channels))
        assert image.data.chunksize == (min(channels, 2), 2, 2)
        np.testing.assert_array_equal(image.data.compute(), source)

    @pytest.mark.parametrize(
        ("samples", "photometric", "extrasamples", "expected"),
        [
            (3, "rgb", None, ["r", "g", "b"]),
            (4, "rgb", ["unassalpha"], ["r", "g", "b", "a"]),
            (4, "rgb", ["unspecified"], [0, 1, 2, 3]),
            (3, "minisblack", None, [0, 1, 2]),
        ],
    )
    def test_samples_are_named_by_the_declared_colour_model(
        self,
        tmp_path: Path,
        samples: int,
        photometric: str,
        extrasamples: list[str] | None,
        expected: list[str | int],
    ) -> None:
        path = tmp_path / "image.tif"
        source = np.arange(30 * samples, dtype=np.uint8).reshape(5, 6, samples)
        tifffile.imwrite(path, source, photometric=photometric, planarconfig="contig", extrasamples=extrasamples)

        image = read_image(ArtifactFile(path), data_axes=None, tiff_series=None, tiff_level=0, raster=DEFAULT_OPTIONS)

        assert isinstance(image, DataArray)
        assert image.coords["c"].to_numpy().tolist() == expected
        np.testing.assert_array_equal(image.data.compute(), source.transpose(2, 0, 1))

    @pytest.mark.parametrize(
        ("photometric", "name", "data_axes", "expected"),
        [
            ("rgb", None, None, ["r", "g", "b"]),
            ("rgb", "H&E", "YXC", ["r", "g", "b"]),
            ("minisblack", None, None, [f"Channel:0:0:sample:{index}" for index in range(3)]),
            ("minisblack", "H&E", None, [f"H&E:sample:{index}" for index in range(3)]),
            ("minisblack", "H&E", "YXC", [f"H&E:sample:{index}" for index in range(3)]),
        ],
    )
    def test_ome_samples_are_named_by_colour_or_channel_record(
        self, tmp_path: Path, photometric: str, name: str | None, data_axes: str | None, expected: list[str]
    ) -> None:
        """OME declares interleaved samples as one channel record; the TIFF declares whether they are RGB."""
        path = tmp_path / "image.ome.tif"
        source = np.arange(90, dtype=np.uint8).reshape(5, 6, 3)
        metadata = {"axes": "YXS"} if name is None else {"axes": "YXS", "Channel": {"Name": [name]}}
        tifffile.imwrite(path, source, photometric=photometric, planarconfig="contig", metadata=metadata)

        image = read_image(
            ArtifactFile(path), data_axes=data_axes, tiff_series=None, tiff_level=0, raster=DEFAULT_OPTIONS
        )

        assert isinstance(image, DataArray)
        assert image.coords["c"].to_numpy().tolist() == expected
        np.testing.assert_array_equal(image.data.compute(), source.transpose(2, 0, 1))

    @pytest.mark.parametrize("names", [["marker-b", "marker-a"], ["same", "same"], [None, None]])
    def test_ome_channel_names_or_unique_ids(self, tmp_path: Path, names: list[str | None]) -> None:
        path = tmp_path / "image.ome.tif"
        source = np.arange(60, dtype=np.uint16).reshape(1, 1, 2, 5, 6)
        tifffile.imwrite(
            path,
            source,
            photometric="minisblack",
            metadata={"axes": "TZCYX", "Channel": {"Name": names}},
        )

        image = read_image(ArtifactFile(path), data_axes=None, tiff_series=None, tiff_level=0, raster=DEFAULT_OPTIONS)

        assert isinstance(image, DataArray)
        assert image.coords["c"].to_numpy().tolist() == (
            names if names == ["marker-b", "marker-a"] else ["Channel:0:0", "Channel:0:1"]
        )
        np.testing.assert_array_equal(image.data.compute(), source[0, 0])

    def test_unknown_stack_requires_explicit_axes(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        source = np.ones((2, 5, 6), dtype=np.uint8)
        tifffile.imwrite(path, source, photometric="minisblack", metadata={"axes": "QYX"})

        with pytest.raises(ValueError, match="Axis 'q'"):
            read_image(ArtifactFile(path), data_axes=None, tiff_series=None, tiff_level=0, raster=DEFAULT_OPTIONS)

        image = read_image(ArtifactFile(path), data_axes="CYX", tiff_series=None, tiff_level=0, raster=DEFAULT_OPTIONS)

        assert isinstance(image, DataArray)
        np.testing.assert_array_equal(image.data.compute(), source)

    def test_default_options_are_single_scale_channel_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        source = np.arange(3 * 40 * 50, dtype=np.uint16).reshape(3, 40, 50)
        tifffile.imwrite(path, source, photometric="minisblack", metadata={"axes": "CYX"}, tile=(16, 16))

        image = read_image(ArtifactFile(path), data_axes=None, tiff_series=None, tiff_level=0, raster=DEFAULT_OPTIONS)

        assert isinstance(image, DataArray)
        # Default tiles exceed the image, so each chunk is one whole channel.
        assert image.data.chunks == ((1, 1, 1), (40,), (50,))
        np.testing.assert_array_equal(image.data.compute(), source)

    def test_scale_factors_build_a_pyramid(self, tmp_path: Path) -> None:
        path = tmp_path / "image.png"
        source = np.arange(3 * 8 * 12, dtype=np.uint8).reshape(8, 12, 3)
        path.write_bytes(imagecodecs.png_encode(source))

        image = read_image(
            ArtifactFile(path), data_axes=None, tiff_series=None, tiff_level=0, raster=make_raster_options((2,), 4)
        )

        assert isinstance(image, DataTree)
        assert list(image.children) == ["scale0", "scale1"]
        scale0 = image["scale0"]["image"]
        assert scale0.data.chunks == ((3,), (4, 4), (4, 4, 4))
        np.testing.assert_array_equal(scale0.data.compute(), source.transpose(2, 0, 1))
        assert image["scale1"]["image"].shape == (3, 4, 6)

    def test_inconsistent_ome_channels_are_rejected(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "image.ome.tif"
        tifffile.imwrite(
            path,
            np.zeros((2, 5, 6), dtype=np.uint16),
            photometric="minisblack",
            metadata={"axes": "CYX", "Channel": {"Name": ["a", "b"]}},
        )

        metadata = inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)
        assert metadata.ome is not None
        metadata.ome.images[0].pixels.channels.pop()
        mocker.patch("spatialdata_io.readers.dbit._images.inspect_tiff", return_value=metadata)

        with pytest.raises(ReaderFormatError, match="1 OME channels for 2 raster channels"):
            read_image(ArtifactFile(path), data_axes=None, tiff_series=None, tiff_level=0, raster=DEFAULT_OPTIONS)

    @pytest.mark.parametrize(("series", "level"), [(1, 0), (0, 1)])
    def test_out_of_range_selection(self, tmp_path: Path, series: int, level: int) -> None:
        path = tmp_path / "image.tif"
        tifffile.imwrite(path, np.ones((5, 6), dtype=np.uint8))

        with pytest.raises(IndexError, match="index") as caught:
            read_image(ArtifactFile(path), data_axes=None, tiff_series=series, tiff_level=level, raster=DEFAULT_OPTIONS)

        context = ReaderErrorContext("dbit", "histology image")
        assert caught.value.__notes__ == [f"While handling {context.describe(path=path)}."]

    def test_several_series_require_a_selection(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.zeros((5, 6), dtype=np.uint8), metadata={"axes": "YX"})
            writer.write(np.full((3, 4), 9, dtype=np.uint16), metadata={"axes": "YX"})

        with pytest.raises(ReaderFormatError, match="found 2 series") as caught:
            read_image(ArtifactFile(path), data_axes=None, tiff_series=None, tiff_level=0, raster=DEFAULT_OPTIONS)
        assert caught.value.expected == "a single series or an explicit tiff_series selection"
        assert caught.value.context == ReaderErrorContext("dbit", "histology image")

        for series, expected in ((0, np.zeros((1, 5, 6), dtype=np.uint8)), (1, np.full((1, 3, 4), 9, np.uint16))):
            image = read_image(
                ArtifactFile(path), data_axes=None, tiff_series=series, tiff_level=0, raster=DEFAULT_OPTIONS
            )
            assert isinstance(image, DataArray)
            assert image.dtype == expected.dtype
            np.testing.assert_array_equal(image.data.compute(), expected)

    def test_scale_factors_cannot_exceed_the_image(self, tmp_path: Path) -> None:
        path = tmp_path / "image.png"
        path.write_bytes(imagecodecs.png_encode(np.zeros((4, 9), dtype=np.uint8)))

        with pytest.raises(ValueError, match=r"exceed the spatial extent of the histology image \(4 x 9 pixels\)"):
            read_image(
                ArtifactFile(path), data_axes=None, tiff_series=None, tiff_level=0, raster=make_raster_options((2, 3))
            )

    @pytest.mark.parametrize("suffix", [".tif", ".png", ".jpg"])
    def test_malformed_image_has_reader_context(self, tmp_path: Path, suffix: str) -> None:
        path = tmp_path / f"image{suffix}"
        path.write_bytes(b"invalid image")

        with pytest.raises(ValueError, match=r"TIFF|PNG|JPEG") as caught:
            read_image(ArtifactFile(path), data_axes=None, tiff_series=None, tiff_level=0, raster=DEFAULT_OPTIONS)

        assert any("dbit" in note and "histology image" in note for note in caught.value.__notes__)

    @pytest.mark.parametrize(("series", "level"), [(0, 0), (None, 1)])
    def test_encoded_images_reject_tiff_options(self, tmp_path: Path, series: int | None, level: int) -> None:
        with pytest.raises(ValueError, match="TIFF-family"):
            read_image(
                ArtifactFile(tmp_path / "image.png"),
                data_axes=None,
                tiff_series=series,
                tiff_level=level,
                raster=DEFAULT_OPTIONS,
            )

    def test_encoded_images_reject_axis_override(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="data_axes"):
            read_image(
                ArtifactFile(tmp_path / "image.png"),
                data_axes="YXC",
                tiff_series=None,
                tiff_level=0,
                raster=DEFAULT_OPTIONS,
            )
