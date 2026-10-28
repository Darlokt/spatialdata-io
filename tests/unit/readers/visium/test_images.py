"""Tests for Visium-owned raster selection and model construction."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import imagecodecs
import numpy as np
import pytest
import tifffile
from dask import array as da
from dask.callbacks import Callback
from xarray import DataArray, DataTree

from spatialdata_io.readers._utils.image import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS, LazyRaster
from spatialdata_io.readers._utils.image.tiff import _dask as tiff_dask
from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile, DatasetRoot
from spatialdata_io.readers.visium._geometry import VisiumTransforms, make_transforms
from spatialdata_io.readers.visium._images import ImageOptions, make_image_options, read_images
from spatialdata_io.readers.visium._layout import H5Counts, VisiumLayout
from spatialdata_io.readers.visium._metadata import ScaleFactors

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from pytest_mock import MockerFixture


def _layout(root: Path, *, fullres: Path | None, hires: Path | None = None, lowres: Path | None = None) -> VisiumLayout:
    placeholder = ArtifactFile(root / "placeholder")
    return VisiumLayout(
        root=DatasetRoot(ArtifactDirectory(root)),
        counts=H5Counts(placeholder, None),
        positions=placeholder,
        position_format="headered",
        scale_factors=placeholder,
        hires_image=None if hires is None else ArtifactFile(hires),
        lowres_image=None if lowres is None else ArtifactFile(lowres),
        fullres_image=None if fullres is None else ArtifactFile(fullres),
    )


def _transforms() -> VisiumTransforms:
    return make_transforms(ScaleFactors(0.5, 0.25, 6.0, {}))


def _options(
    *,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] | Mapping[str, int] = DEFAULT_CHUNKS,
    data_axes: str | None = None,
    tiff_series: int | None = None,
) -> ImageOptions:
    return make_image_options(
        scale_factors=scale_factors,
        chunks=chunks,
        data_axes=data_axes,
        tiff_series=tiff_series,
    )


class TestMakeImageOptions:
    """Normalize borrowed caller options into an immutable canonical value."""

    @pytest.mark.parametrize(
        ("chunks", "expected"),
        [
            (DEFAULT_CHUNKS, {"c": 1, "y": 1_024, "x": 1_024}),
            (8, {"c": 8, "y": 8, "x": 8}),
            ((1, 2, 3), {"c": 1, "y": 2, "x": 3}),
            ({"x": 3, "c": 1, "y": 4}, {"c": 1, "y": 4, "x": 3}),
        ],
    )
    def test_normalizes_chunks(
        self, chunks: int | tuple[int, int, int] | Mapping[str, int], expected: dict[str, int]
    ) -> None:
        result = _options(chunks=chunks)

        assert result.raster.image_chunks() == expected

    @pytest.mark.parametrize(
        ("chunks", "error", "match"),
        [
            ({"y": 2, "x": 2}, ValueError, "keys must match canonical axes"),
            ({"c": 1, "z": 1, "y": 2, "x": 2}, ValueError, "keys must match canonical axes"),
            ({"c": None, "y": 2, "x": 2}, TypeError, "positive integer"),
            ({"c": 1, "y": (1, 1), "x": 2}, TypeError, "positive integer"),
            ((1, (1, 1), 2), TypeError, "positive integer"),
        ],
        ids=["missing-axis", "extra-axis", "none-size", "irregular-mapping", "irregular-tuple"],
    )
    def test_rejects_non_canonical_or_irregular_chunks(
        self, chunks: object, error: type[Exception], match: str
    ) -> None:
        with pytest.raises(error, match=match):
            _options(chunks=chunks)  # pyrefly: ignore[bad-argument-type] - Invalid runtime inputs.

    def test_owns_scale_factors_through_shared_validation(self) -> None:
        factors = [2, 2]

        result = _options(scale_factors=factors)
        factors.append(4)

        assert result.raster.model_scale_factors == [2, 2]
        assert _options().raster.model_scale_factors is None
        with pytest.raises(ValueError, match="at least 2"):
            _options(scale_factors=(1,))

    @pytest.mark.parametrize("tiff_series", [True, -1, 1.5])
    def test_rejects_invalid_tiff_series(self, tiff_series: object) -> None:
        with pytest.raises(ValueError, match="tiff_series"):
            _options(tiff_series=tiff_series)  # pyrefly: ignore[bad-argument-type] - Invalid runtime inputs.

    def test_copies_and_normalizes_explicit_axes(self) -> None:
        axes = ["Z", "C", "Y", "X"]

        result = make_image_options(
            scale_factors=DEFAULT_SCALE_FACTORS, chunks=DEFAULT_CHUNKS, data_axes=axes, tiff_series=1
        )
        axes[0] = "T"

        assert result.data_axes == ("z", "c", "y", "x")
        assert result.tiff_series == 1

    @pytest.mark.parametrize("data_axes", [("YY", "X"), ("X", "X")])
    def test_rejects_invalid_explicit_axes(self, data_axes: tuple[str, str]) -> None:
        with pytest.raises(ValueError, match="data_axes"):
            make_image_options(
                scale_factors=DEFAULT_SCALE_FACTORS, chunks=DEFAULT_CHUNKS, data_axes=data_axes, tiff_series=None
            )


class TestReadImages:
    """Build independent lazy image elements with exact axes and names."""

    def test_reads_tiff_as_lazy_single_scale_cyx(self, tmp_path: Path) -> None:
        path = tmp_path / "full.tif"
        expected = np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7)
        tifffile.imwrite(path, expected, metadata={"axes": "CYX"}, photometric="minisblack", tile=(16, 16))

        images = read_images(
            _layout(tmp_path, fullres=path),
            dataset_id="sample",
            transforms=_transforms(),
            options=_options(chunks=(1, 3, 4)),
        )

        image = images["sample_full_image"]
        assert isinstance(image, DataArray)
        assert image.dims == ("c", "y", "x")
        assert image.shape == expected.shape
        assert image.dtype == expected.dtype
        assert image.chunksizes == {"c": (1, 1, 1), "y": (3, 2), "x": (4, 3)}
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
            transforms=_transforms(),
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

    def test_discovers_hires_and_lowres_independently(self, tmp_path: Path) -> None:
        hires = tmp_path / "hires.png"
        lowres = tmp_path / "lowres.png"
        hires.write_bytes(imagecodecs.png_encode(np.zeros((5, 7, 3), dtype=np.uint8)))
        lowres.write_bytes(imagecodecs.png_encode(np.zeros((2, 4, 3), dtype=np.uint8)))

        images = read_images(
            _layout(tmp_path, fullres=None, hires=hires, lowres=lowres),
            dataset_id="sample",
            transforms=_transforms(),
            options=_options(),
        )

        assert set(images) == {"sample_hires_image", "sample_lowres_image"}
        assert images["sample_hires_image"].shape == (3, 5, 7)
        assert images["sample_lowres_image"].shape == (3, 2, 4)
        # Each PNG frame is one source chunk; the default splits its channels.
        assert images["sample_hires_image"].chunksizes == {"c": (1, 1, 1), "y": (5,), "x": (7,)}

    def test_reads_grayscale_png_and_jpeg_as_single_channel(self, tmp_path: Path) -> None:
        png = tmp_path / "full.png"
        jpeg = tmp_path / "full.jpg"
        values = np.arange(5 * 7, dtype=np.uint8).reshape(5, 7)
        png.write_bytes(imagecodecs.png_encode(values))
        jpeg.write_bytes(imagecodecs.jpeg_encode(values))

        for path in (png, jpeg):
            images = read_images(
                _layout(tmp_path, fullres=path),
                dataset_id="sample",
                transforms=_transforms(),
                options=_options(),
            )

            image = images["sample_full_image"]
            assert isinstance(image, DataArray)
            assert image.shape == (1, 5, 7)

    @pytest.mark.parametrize(
        ("shape", "axes"),
        [((2, 3, 5, 7), ("z", "c", "y", "x")), ((2, 5, 7), ("z", "y", "x"))],
    )
    def test_rejects_non_2d_spatial_raster(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        shape: tuple[int, ...],
        axes: tuple[str, ...],
    ) -> None:
        path = tmp_path / "full.tif"
        path.touch()
        raster = LazyRaster(da.zeros(shape, chunks=shape), axes)
        mocker.patch("spatialdata_io.readers.visium._images._read_raster", return_value=raster)

        with pytest.raises(ValueError, match="no supported spatial or channel meaning"):
            read_images(
                _layout(tmp_path, fullres=path),
                dataset_id="sample",
                transforms=_transforms(),
                options=_options(),
            )

    def test_rejects_unsupported_fullres_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "full.bmp"
        path.touch()

        with pytest.raises(ValueError, match="Unsupported image suffix"):
            read_images(
                _layout(tmp_path, fullres=path),
                dataset_id="sample",
                transforms=_transforms(),
                options=_options(),
            )

    @pytest.mark.parametrize(
        ("data_axes", "tiff_series", "match"),
        [(None, 0, "tiff_series"), ("cyx", None, "data_axes")],
    )
    def test_rejects_tiff_options_for_encoded_images(
        self,
        tmp_path: Path,
        data_axes: str | None,
        tiff_series: int | None,
        match: str,
    ) -> None:
        path = tmp_path / "full.png"
        path.write_bytes(imagecodecs.png_encode(np.zeros((5, 7, 3), dtype=np.uint8)))

        with pytest.raises(ValueError, match=match):
            read_images(
                _layout(tmp_path, fullres=path),
                dataset_id="sample",
                transforms=_transforms(),
                options=_options(data_axes=data_axes, tiff_series=tiff_series),
            )

    @pytest.mark.parametrize(
        ("shape", "axes", "match"),
        [
            ((3, 5, 7), ("y", "x"), "rank"),
            ((3, 5, 7), ("c", "y", "y"), "unique X and Y"),
            ((2, 3, 5, 7), ("c", "s", "y", "x"), "at most one channel"),
        ],
    )
    def test_rejects_invalid_axis_declarations(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        shape: tuple[int, ...],
        axes: tuple[str, ...],
        match: str,
    ) -> None:
        path = tmp_path / "full.tif"
        path.touch()
        raster = LazyRaster(da.zeros(shape, chunks=shape), axes)
        mocker.patch("spatialdata_io.readers.visium._images._read_raster", return_value=raster)

        with pytest.raises(ValueError, match=match):
            read_images(
                _layout(tmp_path, fullres=path),
                dataset_id="sample",
                transforms=_transforms(),
                options=_options(),
            )

    def test_requires_explicit_axes_for_unknown_non_singleton_tiff_axis(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        path = tmp_path / "full.tif"
        path.touch()
        raster = LazyRaster(da.zeros((3, 5, 7), chunks=(3, 5, 7)), ("Q", "Y", "X"))
        mocker.patch("spatialdata_io.readers.visium._images._read_raster", return_value=raster)

        with pytest.raises(ValueError, match="explicit data_axes"):
            read_images(
                _layout(tmp_path, fullres=path),
                dataset_id="sample",
                transforms=_transforms(),
                options=_options(),
            )

        images = read_images(
            _layout(tmp_path, fullres=path),
            dataset_id="sample",
            transforms=_transforms(),
            options=_options(data_axes="cyx"),
        )

        assert images["sample_full_image"].shape == (3, 5, 7)

    def test_drops_named_singleton_dimensions(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "full.tif"
        path.touch()
        raster = LazyRaster(da.zeros((1, 3, 5, 7), chunks=(1, 3, 5, 7)), ("Z", "C", "Y", "X"))
        mocker.patch("spatialdata_io.readers.visium._images._read_raster", return_value=raster)

        images = read_images(
            _layout(tmp_path, fullres=path),
            dataset_id="sample",
            transforms=_transforms(),
            options=_options(),
        )

        assert images["sample_full_image"].shape == (3, 5, 7)

    def test_requires_selection_for_multiple_tiff_series(self, tmp_path: Path) -> None:
        path = tmp_path / "series.tif"
        first = np.arange(35, dtype=np.uint8).reshape(5, 7)
        second = np.arange(99, dtype=np.uint16).reshape(9, 11)
        with tifffile.TiffWriter(path) as writer:
            writer.write(first, photometric="minisblack", metadata={"axes": "YX"})
            writer.write(second, photometric="minisblack", metadata={"axes": "YX"})

        with pytest.raises(ValueError, match="contains 2 series"):
            read_images(
                _layout(tmp_path, fullres=path),
                dataset_id="sample",
                transforms=_transforms(),
                options=_options(),
            )

        images = read_images(
            _layout(tmp_path, fullres=path),
            dataset_id="sample",
            transforms=_transforms(),
            options=_options(tiff_series=1),
        )

        selected = images["sample_full_image"]
        assert selected.dtype == second.dtype
        np.testing.assert_array_equal(selected.data.compute(), second[None])

    @pytest.mark.parametrize("compression", [None, "deflate"])
    def test_btf_is_lazy_serializable_and_partially_read(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        compression: str | None,
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
            images = read_images(
                _layout(tmp_path, fullres=path),
                dataset_id="sample",
                transforms=_transforms(),
                options=_options(),
            )

        assert not executed

        # No pyramid is generated without explicit scale factors, whatever the image size.
        base = images["sample_full_image"]
        assert isinstance(base, DataArray)
        pickle.dumps(base.data.__dask_graph__())

        # Default 1024-pixel chunks span four native tiles along each spatial axis.
        np.testing.assert_array_equal(base.data[0, :8, :8].compute(scheduler="synchronous"), expected[:8, :8])
        assert selection_spy.call_count == 16

    @pytest.mark.parametrize(
        "options",
        [_options(data_axes="cyx"), _options(tiff_series=0), _options(scale_factors=(2,))],
        ids=["data-axes", "tiff-series", "scale-factors"],
    )
    def test_rejects_fullres_options_without_fullres_image(self, tmp_path: Path, options: ImageOptions) -> None:
        with pytest.raises(ValueError, match="require fullres_image_file"):
            read_images(
                _layout(tmp_path, fullres=None),
                dataset_id="sample",
                transforms=_transforms(),
                options=options,
            )
