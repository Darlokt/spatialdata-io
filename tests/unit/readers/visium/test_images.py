"""Tests for Visium-owned raster selection, channel naming and model construction."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import imagecodecs
import numpy as np
import pytest
import tifffile
from dask import array as da
from dask.callbacks import Callback
from spatialdata.transformations import get_transformation
from xarray import DataArray, DataTree

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import LazyRaster
from spatialdata_io.readers._utils.image.tiff import _dask as tiff_dask
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium._geometry import VisiumFrames, make_frames
from spatialdata_io.readers.visium._images import ImageOptions, image_kind, make_image_options, read_images
from spatialdata_io.readers.visium._layout import H5Counts, VisiumLayout
from spatialdata_io.readers.visium._metadata import CountMetadata, ScaleFactors

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytest_mock import MockerFixture

    from spatialdata_io.readers.visium._images import ImageKind

_FULLRES_CONTEXT = ReaderErrorContext("visium", "full-resolution image", "2.1.0")


def _layout(root: Path, *, fullres: Path | None, hires: Path | None = None, lowres: Path | None = None) -> VisiumLayout:
    placeholder = ArtifactFile(root / "placeholder")
    return VisiumLayout(
        counts=H5Counts(placeholder, None, CountMetadata(None, None, "spaceranger-2.1.0")),
        positions=placeholder,
        position_format="headered",
        scale_factors=placeholder,
        hires_image=None if hires is None else ArtifactFile(hires),
        lowres_image=None if lowres is None else ArtifactFile(lowres),
        fullres_image=None if fullres is None else ArtifactFile(fullres),
    )


def _frames() -> VisiumFrames:
    return make_frames("sample", ScaleFactors(0.5, 0.25, 6.0, {}))


def _options(
    kind: ImageKind | None = "tiff",
    *,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
    data_axes: str | None = None,
    tiff_series: int | None = None,
) -> ImageOptions:
    return make_image_options(
        kind, data_axes=data_axes, tiff_series=tiff_series, scale_factors=scale_factors, chunks=chunks
    )


def _read(root: Path, fullres: Path, options: ImageOptions | None = None) -> DataArray | DataTree:
    images = read_images(
        _layout(root, fullres=fullres),
        dataset_id="sample",
        frames=_frames(),
        options=_options(image_kind(fullres)) if options is None else options,
    )
    return images["sample_full_image"]


def _channels(image: DataArray | DataTree) -> list[object]:
    return image.coords["c"].to_numpy().tolist()


def _fake_tiff(root: Path, mocker: MockerFixture, shape: tuple[int, ...], axes: tuple[str, ...]) -> Path:
    """Return a real non-OME TIFF whose pixel reader yields a raster with ``axes``."""
    path = root / "full.tif"
    tifffile.imwrite(path, np.zeros((2, 2), dtype=np.uint8))
    raster = LazyRaster(da.from_array(np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)), axes)
    mocker.patch("spatialdata_io.readers.visium._images.read_tiff", return_value=raster)
    return path


class TestImageKind:
    """Select the decoder from the caller's suffix before any I/O."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("full.tif", "tiff"),
            ("full.TIFF", "tiff"),
            ("full.btf", "tiff"),
            ("full.ome.tif", "tiff"),
            ("full.png", "png"),
            ("full.JPG", "jpeg"),
            ("full.jpeg", "jpeg"),
        ],
    )
    def test_maps_suffixes(self, name: str, expected: str) -> None:
        assert image_kind(Path("dir") / name) == expected
        assert image_kind(name) == expected

    @pytest.mark.parametrize("name", ["full.bmp", "full", "full.tif.gz"])
    def test_rejects_unsupported_suffixes(self, name: str) -> None:
        with pytest.raises(ValueError, match="fullres_image_file must be a TIFF/BigTIFF"):
            image_kind(name)


class TestMakeImageOptions:
    """Validate borrowed caller options into one immutable value."""

    @pytest.mark.parametrize(
        ("chunks", "expected"),
        [
            (DEFAULT_CHUNKS, {"c": 1, "y": 1_024, "x": 1_024}),
            (8, {"c": 8, "y": 8, "x": 8}),
            ((1, 2, 3), {"c": 1, "y": 2, "x": 3}),
        ],
    )
    def test_normalizes_chunks(self, chunks: int | tuple[int, int, int], expected: dict[str, int]) -> None:
        assert _options(chunks=chunks).raster.image_chunks() == expected

    @pytest.mark.parametrize("chunks", [{"c": 1, "y": 2, "x": 3}, (1, (1, 1), 2), [1, 2, 3]])
    def test_rejects_mapping_and_irregular_chunks(self, chunks: object) -> None:
        with pytest.raises(TypeError, match="chunks must be a positive integer"):
            _options(chunks=chunks)  # pyrefly: ignore[bad-argument-type] - Invalid runtime inputs.

    def test_owns_scale_factors_through_shared_validation(self) -> None:
        factors = [2, 2]

        result = _options(scale_factors=factors)
        factors.append(4)

        assert result.raster.model_scale_factors == [2, 2]
        assert _options().raster.model_scale_factors is None
        with pytest.raises(ValueError, match="at least 2"):
            _options(scale_factors=(1,))

    @pytest.mark.parametrize(
        ("tiff_series", "error", "match"),
        [
            (True, TypeError, "tiff_series must be an integer or None; found bool"),
            (1.5, TypeError, "tiff_series must be an integer or None; found float"),
            ("1", TypeError, "tiff_series must be an integer or None; found str"),
            (-1, ValueError, "tiff_series must be nonnegative; found -1"),
        ],
    )
    def test_rejects_invalid_tiff_series(self, tiff_series: object, error: type[Exception], match: str) -> None:
        with pytest.raises(error, match=match):
            _options(tiff_series=tiff_series)  # pyrefly: ignore[bad-argument-type] - Invalid runtime inputs.

    @pytest.mark.parametrize(("data_axes", "expected"), [("ZCYX", "zcyx"), ("yxs", "yxs"), ("Cyx", "cyx")])
    def test_normalizes_data_axes_case(self, data_axes: str, expected: str) -> None:
        result = _options(data_axes=data_axes, tiff_series=1)

        assert result.data_axes == expected
        assert result.tiff_series == 1

    @pytest.mark.parametrize("data_axes", [("C", "Y", "X"), ["C", "Y", "X"], b"CYX"])
    def test_rejects_non_string_data_axes(self, data_axes: object) -> None:
        with pytest.raises(TypeError, match="data_axes must be a string such as 'CYX'"):
            _options(data_axes=data_axes)  # pyrefly: ignore[bad-argument-type] - Invalid runtime inputs.

    @pytest.mark.parametrize("data_axes", ["YYX", "CSYX", "ZY", "C1YX", "ÄYX", "", "C Y X"])
    def test_rejects_invalid_data_axes(self, data_axes: str) -> None:
        with pytest.raises(ValueError, match=f"data_axes must contain distinct .*found {data_axes!r}"):
            _options(data_axes=data_axes)

    @pytest.mark.parametrize("kind", [None, "png", "jpeg"])
    @pytest.mark.parametrize(("data_axes", "tiff_series"), [("CYX", None), (None, 0)])
    def test_tiff_options_require_tiff(
        self, kind: ImageKind | None, data_axes: str | None, tiff_series: int | None
    ) -> None:
        with pytest.raises(ValueError, match="require a TIFF-family fullres_image_file"):
            _options(kind, data_axes=data_axes, tiff_series=tiff_series)

    def test_scale_factors_require_fullres_image(self) -> None:
        with pytest.raises(ValueError, match="scale_factors require fullres_image_file"):
            _options(None, scale_factors=(2,))

        assert _options("png", scale_factors=(2,)).raster.scale_factors == (2,)
        assert _options(None, chunks=(1, 8, 8)).raster.chunks == (1, 8, 8)


class TestReadImages:
    """Build independent lazy image elements with exact axes, names and frames."""

    def test_reads_tiff_as_lazy_single_scale_cyx(self, tmp_path: Path) -> None:
        path = tmp_path / "full.tif"
        expected = np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7)
        tifffile.imwrite(path, expected, metadata={"axes": "CYX"}, photometric="minisblack", tile=(16, 16))

        image = _read(tmp_path, path, _options(chunks=(1, 3, 4)))

        assert isinstance(image, DataArray)
        assert image.dims == ("c", "y", "x")
        assert image.shape == expected.shape
        assert image.dtype == expected.dtype
        assert image.chunksizes == {"c": (1, 1, 1), "y": (3, 2), "x": (4, 3)}
        transformations = get_transformation(image, get_all=True)
        assert isinstance(transformations, dict)
        assert set(transformations) == {"sample_pixels"}
        np.testing.assert_array_equal(image.data.compute(), expected)

    def test_scale_factors_pyramid_only_the_fullres_image(self, tmp_path: Path) -> None:
        fullres = tmp_path / "full.tif"
        hires = tmp_path / "hires.png"
        expected = np.arange(3 * 16 * 20, dtype=np.uint16).reshape(3, 16, 20)
        tifffile.imwrite(fullres, expected, metadata={"axes": "CYX"}, photometric="minisblack", tile=(16, 16))
        hires.write_bytes(imagecodecs.png_encode(np.zeros((8, 10, 3), dtype=np.uint8)))

        images = read_images(
            _layout(tmp_path, fullres=fullres, hires=hires),
            dataset_id="sample",
            frames=_frames(),
            options=_options(scale_factors=(2, 2), chunks=(1, 4, 4)),
        )

        pyramid = images["sample_full_image"]
        assert isinstance(pyramid, DataTree)
        assert list(pyramid.children) == ["scale0", "scale1", "scale2"]
        assert [pyramid[f"scale{index}"]["image"].shape for index in range(3)] == [
            (3, 16, 20),
            (3, 8, 10),
            (3, 4, 5),
        ]
        scale0 = pyramid["scale0"]["image"]
        assert scale0.chunksizes == {"c": (1, 1, 1), "y": (4, 4, 4, 4), "x": (4, 4, 4, 4, 4)}
        np.testing.assert_array_equal(scale0.data.compute(), expected)
        # Vendor-downsampled PNGs are already reduced images.
        assert isinstance(images["sample_hires_image"], DataArray)

    def test_rejects_scale_factors_beyond_the_image_extent(self, tmp_path: Path) -> None:
        path = tmp_path / "full.tif"
        tifffile.imwrite(path, np.zeros((5, 7), dtype=np.uint8))

        with pytest.raises(
            ValueError, match=r"scale factors \(2, 4\) exceed the spatial extent of the full-resolution"
        ):
            _read(tmp_path, path, _options(scale_factors=(2, 4)))

    def test_reads_hires_and_lowres_independently(self, tmp_path: Path) -> None:
        hires = tmp_path / "hires.png"
        lowres = tmp_path / "lowres.png"
        hires.write_bytes(imagecodecs.png_encode(np.zeros((5, 7, 3), dtype=np.uint8)))
        lowres.write_bytes(imagecodecs.png_encode(np.zeros((2, 4, 3), dtype=np.uint8)))

        images = read_images(
            _layout(tmp_path, fullres=None, hires=hires, lowres=lowres),
            dataset_id="sample",
            frames=_frames(),
            options=_options(None),
        )

        assert set(images) == {"sample_hires_image", "sample_lowres_image"}
        assert images["sample_hires_image"].shape == (3, 5, 7)
        assert images["sample_lowres_image"].shape == (3, 2, 4)
        # Each PNG frame is one source chunk; the default splits its channels.
        assert images["sample_hires_image"].chunksizes == {"c": (1, 1, 1), "y": (5,), "x": (7,)}
        for name, scale in (("hires", 2.0), ("lowres", 4.0)):
            transformations = get_transformation(images[f"sample_{name}_image"], get_all=True)
            assert isinstance(transformations, dict)
            assert set(transformations) == {f"sample_{name}_pixels", "sample_pixels"}
            np.testing.assert_array_equal(
                transformations["sample_pixels"].to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
                np.diag([scale, scale, 1.0]),
            )

    @pytest.mark.parametrize(
        ("shape", "expected"),
        [((5, 7), [0]), ((5, 7, 2), [0, 1]), ((5, 7, 3), ["r", "g", "b"]), ((5, 7, 4), ["r", "g", "b", "a"])],
        ids=["gray", "gray-alpha", "rgb", "rgba"],
    )
    def test_png_channels_follow_the_declared_colour_type(
        self, tmp_path: Path, shape: tuple[int, ...], expected: list[object]
    ) -> None:
        hires = tmp_path / "hires.png"
        hires.write_bytes(imagecodecs.png_encode(np.zeros(shape, dtype=np.uint8)))

        images = read_images(
            _layout(tmp_path, fullres=None, hires=hires),
            dataset_id="sample",
            frames=_frames(),
            options=_options(None),
        )

        assert _channels(images["sample_hires_image"]) == expected

    def test_reads_grayscale_png_and_jpeg_as_single_channel(self, tmp_path: Path) -> None:
        png = tmp_path / "full.png"
        jpeg = tmp_path / "full.jpg"
        values = np.arange(5 * 7, dtype=np.uint8).reshape(5, 7)
        png.write_bytes(imagecodecs.png_encode(values))
        jpeg.write_bytes(imagecodecs.jpeg_encode(values))

        for path in (png, jpeg):
            image = _read(tmp_path, path)

            assert isinstance(image, DataArray)
            assert image.shape == (1, 5, 7)
            assert _channels(image) == [0]

    def test_rgb_jpeg_is_named_rgb(self, tmp_path: Path) -> None:
        path = tmp_path / "full.jpg"
        path.write_bytes(imagecodecs.jpeg_encode(np.zeros((5, 7, 3), dtype=np.uint8)))

        assert _channels(_read(tmp_path, path)) == ["r", "g", "b"]

    @pytest.mark.parametrize(
        ("photometric", "extrasamples", "ome", "expected"),
        [
            ("rgb", None, False, ["r", "g", "b"]),
            ("rgb", ("unassalpha",), False, ["r", "g", "b", "a"]),
            ("minisblack", None, False, [0, 1, 2]),
            ("rgb", None, True, ["r", "g", "b"]),
        ],
        ids=["rgb", "rgba", "minisblack-samples", "ome-rgb"],
    )
    def test_tiff_samples_are_rgb_only_when_declared(
        self,
        tmp_path: Path,
        photometric: Literal["rgb", "minisblack"],
        extrasamples: tuple[str, ...] | None,
        expected: list[object],
        *,
        ome: bool,
    ) -> None:
        path = tmp_path / "full.tif"
        samples = 3 if extrasamples is None else 4
        tifffile.imwrite(
            path,
            np.zeros((5, 7, samples), dtype=np.uint8),
            photometric=photometric,
            planarconfig="contig",
            extrasamples=extrasamples,
            ome=ome,
        )

        assert _channels(_read(tmp_path, path)) == expected

    @pytest.mark.parametrize("ome", [True, False])
    def test_four_channel_fluorescence_is_never_rgba(self, tmp_path: Path, *, ome: bool) -> None:
        path = tmp_path / "full.ome.tif"
        names = ["DAPI", "CD3", "CD8", "PanCK"]
        values = np.arange(4 * 5 * 7, dtype=np.uint16).reshape(4, 5, 7)
        metadata: dict[str, object] = {"axes": "CYX"}
        if ome:
            metadata["Channel"] = {"Name": names}
        tifffile.imwrite(path, values, ome=ome, photometric="minisblack", metadata=metadata)

        image = _read(tmp_path, path)

        assert _channels(image) == (names if ome else [0, 1, 2, 3])
        assert image.dtype == np.dtype(np.uint16)
        np.testing.assert_array_equal(image.data.compute(), values)

    def test_single_channel_ome_tiff_keeps_its_name(self, tmp_path: Path) -> None:
        path = tmp_path / "full.ome.tif"
        tifffile.imwrite(
            path, np.zeros((5, 7), dtype=np.uint16), ome=True, metadata={"axes": "YX", "Channel": {"Name": ["DAPI"]}}
        )

        assert _channels(_read(tmp_path, path)) == ["DAPI"]

    def test_axes_reinterpreted_as_channels_get_positions(self, tmp_path: Path) -> None:
        path = tmp_path / "full.ome.tif"
        tifffile.imwrite(path, np.zeros((3, 5, 7), dtype=np.uint16), ome=True, metadata={"axes": "ZYX"})

        image = _read(tmp_path, path, _options(data_axes="CYX"))

        assert image.shape == (3, 5, 7)
        assert _channels(image) == [0, 1, 2]

    # The shared CYX normalization tests every axis permutation; these check the reader's context.
    def test_unmappable_declared_axes_are_malformed_content(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _fake_tiff(tmp_path, mocker, (2, 5, 7), ("z", "y", "x"))

        with pytest.raises(ReaderFormatError, match="absent from target_axes") as caught:
            _read(tmp_path, path)

        assert caught.value.context == _FULLRES_CONTEXT
        assert caught.value.path == path
        assert type(caught.value.__cause__) is ValueError

    def test_unmappable_data_axes_are_a_caller_error_with_context(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _fake_tiff(tmp_path, mocker, (3, 5, 7), ("c", "y", "x"))

        with pytest.raises(ValueError, match="rank") as caught:
            _read(tmp_path, path, _options(data_axes="YX"))

        assert type(caught.value) is ValueError
        assert caught.value.__notes__ == [f"While handling {_FULLRES_CONTEXT.describe(path=path)}."]

    def test_data_axes_resolve_an_unknown_axis(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _fake_tiff(tmp_path, mocker, (3, 5, 7), ("Q", "Y", "X"))

        image = _read(tmp_path, path, _options(data_axes="cyx"))

        assert image.shape == (3, 5, 7)
        assert _channels(image) == [0, 1, 2]

    @pytest.mark.parametrize("axes", [("Z", "C", "Y", "X"), ("Z", "S", "Y", "X")])
    def test_drops_named_singleton_dimensions(
        self, tmp_path: Path, mocker: MockerFixture, axes: tuple[str, ...]
    ) -> None:
        path = _fake_tiff(tmp_path, mocker, (1, 3, 5, 7), axes)

        image = _read(tmp_path, path)

        assert image.shape == (3, 5, 7)
        np.testing.assert_array_equal(image.data.compute(), np.arange(3 * 5 * 7).reshape(3, 5, 7))

    def test_requires_selection_for_multiple_tiff_series(self, tmp_path: Path) -> None:
        path = tmp_path / "series.tif"
        first = np.arange(35, dtype=np.uint8).reshape(5, 7)
        second = np.arange(99, dtype=np.uint16).reshape(9, 11)
        with tifffile.TiffWriter(path) as writer:
            writer.write(first, photometric="minisblack", metadata={"axes": "YX"})
            writer.write(second, photometric="minisblack", metadata={"axes": "YX"})

        with pytest.raises(ReaderFormatError, match="2 TIFF series") as caught:
            _read(tmp_path, path)
        assert caught.value.context == _FULLRES_CONTEXT
        assert caught.value.path == path
        assert caught.value.expected == "one TIFF series or an explicit tiff_series selection"

        selected = _read(tmp_path, path, _options(tiff_series=1))
        assert selected.dtype == second.dtype
        np.testing.assert_array_equal(selected.data.compute(), second[None])

    def test_out_of_range_series_names_the_image(self, tmp_path: Path) -> None:
        path = tmp_path / "full.tif"
        tifffile.imwrite(path, np.zeros((5, 7), dtype=np.uint8))

        with pytest.raises(IndexError) as caught:
            _read(tmp_path, path, _options(tiff_series=1))

        assert any("full-resolution image" in note for note in caught.value.__notes__)

    @pytest.mark.parametrize("compression", [None, "deflate"])
    def test_btf_is_lazy_serializable_and_partially_read(
        self, tmp_path: Path, mocker: MockerFixture, compression: str | None
    ) -> None:
        path = tmp_path / "full.btf"
        expected = np.arange(1_100 * 1_100, dtype=np.uint16).reshape(1_100, 1_100)
        tifffile.imwrite(
            path,
            expected,
            compression=compression,
            photometric="minisblack",
            metadata={"axes": "YX"},
            tile=(256, 256),
            bigtiff=True,
        )
        executed: list[object] = []
        selection_spy = mocker.spy(tiff_dask, "_read_tiff_selection")

        with Callback(pretask=lambda key, *_: executed.append(key)):
            base = _read(tmp_path, path)

        assert not executed
        # No pyramid is generated without explicit scale factors, whatever the image size.
        assert isinstance(base, DataArray)
        pickle.dumps(base.data.__dask_graph__())

        # Default 1024-pixel chunks span four native tiles along each spatial axis.
        np.testing.assert_array_equal(base.data[0, :8, :8].compute(scheduler="synchronous"), expected[:8, :8])
        assert selection_spy.call_count == 16
