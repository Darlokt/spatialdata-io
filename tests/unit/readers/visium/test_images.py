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
from xarray import DataTree

from spatialdata_io.readers._utils.image import LazyRaster
from spatialdata_io.readers._utils.image.tiff import _dask as tiff_dask
from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile, DatasetRoot
from spatialdata_io.readers.visium._geometry import VisiumTransforms, make_transforms
from spatialdata_io.readers.visium._images import Chunks, ImageOptions, make_image_options, read_images
from spatialdata_io.readers.visium._layout import H5Counts, VisiumLayout
from spatialdata_io.readers.visium._metadata import ScaleFactors

if TYPE_CHECKING:
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
    chunks: Chunks | None = None,
    *,
    data_axes: str | None = None,
    tiff_series: int | None = None,
) -> ImageOptions:
    return make_image_options(
        chunks=chunks,
        data_axes=data_axes,
        tiff_series=tiff_series,
    )


class TestMakeImageOptions:
    """Normalize borrowed caller options into an immutable canonical value."""

    @pytest.mark.parametrize(
        ("chunks", "expected"),
        [
            (None, (None, 1_024, 1_024)),
            (8, (8, 8, 8)),
            ((1, 2, 3), (1, 2, 3)),
            ({"c": None, "y": 4, "x": (3, 2)}, (None, 4, (3, 2))),
        ],
    )
    def test_normalizes_chunks(self, chunks: Chunks | None, expected: object) -> None:
        result = _options(chunks)

        assert result.chunks == expected

    @pytest.mark.parametrize(
        "chunks",
        [True, 0, "invalid", (1, 2), {"y": 2, "x": 2}, (True, 2, 2), ((), 2, 2), ((-1,), 2, 2)],
    )
    def test_rejects_invalid_chunks(self, chunks: object) -> None:
        with pytest.raises((TypeError, ValueError), match="chunks"):
            _options(chunks)  # type: ignore[arg-type]  # intentionally exercise the runtime boundary

    @pytest.mark.parametrize("tiff_series", [True, -1, 1.5])
    def test_rejects_invalid_tiff_series(self, tiff_series: object) -> None:
        with pytest.raises(ValueError, match="tiff_series"):
            make_image_options(chunks=None, data_axes=None, tiff_series=tiff_series)  # type: ignore[arg-type]

    def test_copies_and_normalizes_explicit_axes(self) -> None:
        axes = ["Z", "C", "Y", "X"]

        result = make_image_options(chunks=None, data_axes=axes, tiff_series=1)
        axes[0] = "T"

        assert result.data_axes == ("z", "c", "y", "x")
        assert result.tiff_series == 1

    @pytest.mark.parametrize("data_axes", [("YY", "X"), ("X", "X")])
    def test_rejects_invalid_explicit_axes(self, data_axes: tuple[str, str]) -> None:
        with pytest.raises(ValueError, match="data_axes"):
            make_image_options(chunks=None, data_axes=data_axes, tiff_series=None)


class TestReadImages:
    """Build independent lazy image elements with exact axes and names."""

    def test_reads_tiff_as_lazy_cyx(self, tmp_path: Path) -> None:
        path = tmp_path / "full.tif"
        expected = np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7)
        tifffile.imwrite(path, expected, metadata={"axes": "CYX"}, photometric="minisblack", tile=(16, 16))

        images = read_images(
            _layout(tmp_path, fullres=path),
            dataset_id="sample",
            transforms=_transforms(),
            options=make_image_options(chunks=(1, 3, 4), data_axes=None, tiff_series=None),
        )

        pyramid = images["sample_full_image"]
        assert isinstance(pyramid, DataTree)
        assert list(pyramid.children) == ["scale0", "scale1", "scale2"]
        image = pyramid["scale0"]["image"]
        assert image.dims == ("c", "y", "x")
        assert image.shape == expected.shape
        assert image.dtype == expected.dtype
        np.testing.assert_array_equal(image.data[:, :2, :3].compute(), expected[:, :2, :3])

    def test_discovers_hires_and_lowres_independently(self, tmp_path: Path) -> None:
        hires = tmp_path / "hires.png"
        lowres = tmp_path / "lowres.png"
        hires.write_bytes(imagecodecs.png_encode(np.zeros((5, 7, 3), dtype=np.uint8)))
        lowres.write_bytes(imagecodecs.png_encode(np.zeros((2, 4, 3), dtype=np.uint8)))

        images = read_images(
            _layout(tmp_path, fullres=None, hires=hires, lowres=lowres),
            dataset_id="sample",
            transforms=_transforms(),
            options=make_image_options(chunks=None, data_axes=None, tiff_series=None),
        )

        assert set(images) == {"sample_hires_image", "sample_lowres_image"}
        assert images["sample_hires_image"].shape == (3, 5, 7)
        assert images["sample_lowres_image"].shape == (3, 2, 4)

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
                options=make_image_options(chunks=None, data_axes=None, tiff_series=None),
            )
            pyramid = images["sample_full_image"]
            assert isinstance(pyramid, DataTree)
            assert pyramid["scale0"]["image"].shape == (1, 5, 7)

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
                options=make_image_options(chunks=None, data_axes=None, tiff_series=None),
            )

    def test_rejects_unsupported_fullres_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "full.bmp"
        path.touch()

        with pytest.raises(ValueError, match="Unsupported image suffix"):
            read_images(
                _layout(tmp_path, fullres=path),
                dataset_id="sample",
                transforms=_transforms(),
                options=make_image_options(chunks=None, data_axes=None, tiff_series=None),
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
                options=make_image_options(chunks=None, data_axes=data_axes, tiff_series=tiff_series),
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
                options=make_image_options(chunks=None, data_axes=None, tiff_series=None),
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
                options=make_image_options(chunks=None, data_axes=None, tiff_series=None),
            )

        images = read_images(
            _layout(tmp_path, fullres=path),
            dataset_id="sample",
            transforms=_transforms(),
            options=make_image_options(chunks=None, data_axes="cyx", tiff_series=None),
        )
        assert images["sample_full_image"]["scale0"]["image"].shape == (3, 5, 7)

    def test_drops_named_singleton_dimensions(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "full.tif"
        path.touch()
        raster = LazyRaster(da.zeros((1, 3, 5, 7), chunks=(1, 3, 5, 7)), ("Z", "C", "Y", "X"))
        mocker.patch("spatialdata_io.readers.visium._images._read_raster", return_value=raster)

        images = read_images(
            _layout(tmp_path, fullres=path),
            dataset_id="sample",
            transforms=_transforms(),
            options=make_image_options(chunks=None, data_axes=None, tiff_series=None),
        )

        assert images["sample_full_image"]["scale0"]["image"].shape == (3, 5, 7)

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
                options=make_image_options(chunks=None, data_axes=None, tiff_series=None),
            )

        images = read_images(
            _layout(tmp_path, fullres=path),
            dataset_id="sample",
            transforms=_transforms(),
            options=make_image_options(chunks=None, data_axes=None, tiff_series=1),
        )
        selected = images["sample_full_image"]["scale0"]["image"]
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
                options=make_image_options(chunks=None, data_axes=None, tiff_series=None),
            )

        assert not executed
        base = images["sample_full_image"]["scale0"]["image"]
        pickle.dumps(base.data.__dask_graph__())
        np.testing.assert_array_equal(base.data[0, :8, :8].compute(scheduler="synchronous"), expected[:8, :8])
        assert selection_spy.call_count == 16

    def test_rejects_fullres_options_without_fullres_image(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="require fullres_image_file"):
            read_images(
                _layout(tmp_path, fullres=None),
                dataset_id="sample",
                transforms=_transforms(),
                options=make_image_options(chunks=None, data_axes="cyx", tiff_series=None),
            )
