"""Tests for normalizing a 2D raster to CYX from declared or caller axes, with its channel names."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pytest
import tifffile
from dask import delayed

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import LazyRaster, inspect_tiff, normalize_cyx_image, read_tiff
from spatialdata_io.readers._utils.paths import ArtifactFile

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.image import TiffMetadata
    from spatialdata_io.readers._utils.image._types import SampleColor

_CONTEXT = ReaderErrorContext("test", "image", "1.0")
_PATH = Path("image.tif")
_EXPECTED = "Y and X axes with at most one channel or samples axis and only singleton other axes"


def _raster(shape: tuple[int, ...], axes: str, color: SampleColor | None = None) -> LazyRaster:
    data = da.from_array(np.arange(np.prod(shape), dtype=np.uint16).reshape(shape), chunks=shape)
    return LazyRaster(data, tuple(axes), color)


def _tiff(
    path: Path,
    data: np.ndarray,
    *,
    metadata: dict[str, object],
    photometric: str,
    planarconfig: str | None = None,
) -> tuple[LazyRaster, tuple[TiffMetadata, int]]:
    """Write an OME-TIFF and return its lazy base raster with the inspected metadata and series."""
    tifffile.imwrite(path, data, ome=True, photometric=photometric, planarconfig=planarconfig, metadata=metadata)
    inspected = inspect_tiff(ArtifactFile(path), context=_CONTEXT)
    return read_tiff(inspected, series=0, level=0, context=_CONTEXT), (inspected, 0)


class TestNormalizeCyxImage:
    """Map declared or caller axes to CYX and name channels by colour, OME records or position."""

    @pytest.mark.parametrize(
        ("shape", "axes", "data_axes", "order", "added"),
        [
            ((4, 5), "YX", None, (0, 1), True),
            ((5, 4), "XY", None, (1, 0), True),
            ((3, 4, 5), "CYX", None, (0, 1, 2), False),
            ((4, 5, 3), "YXC", None, (2, 0, 1), False),
            ((4, 5, 3), "YXS", None, (2, 0, 1), False),
            ((3, 4, 5), "SYX", None, (0, 1, 2), False),
            ((3, 4, 5), "QYX", "cyx", (0, 1, 2), False),
            ((3, 4, 5), "ZYX", "syx", (0, 1, 2), False),
            ((4, 5, 3), "CYX", "yxc", (2, 0, 1), False),
            ((5, 4), "YX", "xy", (1, 0), True),
        ],
    )
    def test_maps_samples_to_channels_and_adds_a_missing_channel(  # noqa: PLR0913, RUF100 - One case per row.
        self, shape: tuple[int, ...], axes: str, data_axes: str | None, order: tuple[int, ...], *, added: bool
    ) -> None:
        raster = _raster(shape, axes)
        expected = raster.data.compute().transpose(order)

        data, channels = normalize_cyx_image(raster, data_axes, tiff=None, context=_CONTEXT, path=_PATH)

        np.testing.assert_array_equal(data.compute(), expected[None] if added else expected)
        assert data.dtype == np.uint16
        assert channels is None

    @pytest.mark.parametrize(("shape", "axes"), [((1, 3, 4, 5), "ZCYX"), ((1, 4, 5), "TYX"), ((4, 1, 5), "YSX")])
    def test_drops_named_singletons(self, shape: tuple[int, ...], axes: str) -> None:
        raster = _raster(shape, axes)

        data, _ = normalize_cyx_image(raster, None, tiff=None, context=_CONTEXT, path=_PATH)

        np.testing.assert_array_equal(data.compute(), raster.data.compute().reshape(data.shape))
        assert data.ndim == 3

    def test_accepts_validated_axes_in_any_letter_case(self) -> None:
        raster = _raster((4, 5, 3), "YXQ")

        data, _ = normalize_cyx_image(raster, "YXc", tiff=None, context=_CONTEXT, path=_PATH)

        assert data.shape == (3, 4, 5)

    def test_reads_no_pixels(self) -> None:
        @delayed
        def fail() -> np.ndarray:
            message = "normalization computed pixels"
            raise AssertionError(message)

        raster = LazyRaster(da.from_delayed(fail(), shape=(4, 5, 3), dtype=np.uint8), ("Y", "X", "S"), "rgb")

        data, channels = normalize_cyx_image(raster, None, tiff=None, context=_CONTEXT, path=_PATH)

        assert data.shape == (3, 4, 5)
        assert channels == ["r", "g", "b"]

    @pytest.mark.parametrize(
        ("color", "data_axes", "expected"),
        [
            ("rgb", None, ["r", "g", "b"]),
            ("rgba", None, ["r", "g", "b", "a"]),
            ("rgb", "yxc", ["r", "g", "b"]),
            (None, None, None),
            # The colour model describes the samples, not an axis read as channels instead.
            ("rgb", "ycx", None),
        ],
    )
    def test_colour_names_only_the_declared_samples(
        self, color: SampleColor | None, data_axes: str | None, expected: list[str] | None
    ) -> None:
        samples = 4 if color == "rgba" else 3
        raster = _raster((4, 5, samples), "YXS", color)

        _, channels = normalize_cyx_image(raster, data_axes, tiff=None, context=_CONTEXT, path=_PATH)

        assert channels == expected

    @pytest.mark.parametrize(
        ("data", "metadata", "data_axes", "expected"),
        [
            (
                np.zeros((2, 4, 5), np.uint16),
                {"axes": "CYX", "Channel": {"Name": ["DAPI", "CD3"]}},
                None,
                ["DAPI", "CD3"],
            ),
            (
                np.zeros((2, 4, 5), np.uint16),
                {"axes": "CYX", "Channel": {"Name": ["A", "A"]}},
                None,
                ["Channel:0:0", "Channel:0:1"],
            ),
            (np.zeros((4, 5), np.uint16), {"axes": "YX", "Channel": {"Name": ["DAPI"]}}, None, ["DAPI"]),
            (np.zeros((4, 5), np.uint16), {"axes": "YX", "Channel": {"Name": ["DAPI"]}}, "xy", ["DAPI"]),
            # One record describes the single channel, not the Z planes or columns read as channels.
            (np.zeros((3, 4, 5), np.uint16), {"axes": "ZYX", "Channel": {"Name": ["DAPI"]}}, "cyx", None),
            (np.zeros((3, 4, 5), np.uint16), {"axes": "ZYX", "Channel": {"Name": ["DAPI"]}}, "syx", None),
            (np.zeros((2, 4, 5), np.uint16), {"axes": "CYX", "Channel": {"Name": ["DAPI", "CD3"]}}, "ycx", None),
        ],
    )
    def test_ome_records_name_only_the_axis_they_describe(
        self,
        tmp_path: Path,
        data: np.ndarray,
        metadata: dict[str, object],
        data_axes: str | None,
        expected: list[str] | None,
    ) -> None:
        raster, tiff = _tiff(tmp_path / "image.ome.tif", data, photometric="minisblack", metadata=metadata)

        _, channels = normalize_cyx_image(raster, data_axes, tiff=tiff, context=_CONTEXT, path=_PATH)

        assert channels == expected

    @pytest.mark.parametrize(
        ("photometric", "expected"),
        [("rgb", ["r", "g", "b"]), ("minisblack", ["H&E:sample:0", "H&E:sample:1", "H&E:sample:2"])],
    )
    def test_ome_samples_yield_to_a_declared_colour_model(
        self, tmp_path: Path, photometric: str, expected: list[str]
    ) -> None:
        raster, tiff = _tiff(
            tmp_path / "image.ome.tif",
            np.zeros((4, 5, 3), np.uint8),
            photometric=photometric,
            planarconfig="contig",
            metadata={"axes": "YXS", "Channel": {"Name": ["H&E"]}},
        )

        _, channels = normalize_cyx_image(raster, None, tiff=tiff, context=_CONTEXT, path=_PATH)

        assert channels == expected

    def test_non_ome_tiff_channels_are_positional(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        tifffile.imwrite(path, np.zeros((2, 4, 5), np.uint16), photometric="minisblack", metadata={"axes": "CYX"})
        metadata = inspect_tiff(ArtifactFile(path), context=_CONTEXT)
        raster = read_tiff(metadata, series=0, level=0, context=_CONTEXT)

        assert normalize_cyx_image(raster, None, tiff=(metadata, 0), context=_CONTEXT, path=path)[1] is None

    def test_corrupt_corresponding_ome_records_fail(self, tmp_path: Path) -> None:
        raster, (metadata, series) = _tiff(
            tmp_path / "image.ome.tif",
            np.zeros((2, 4, 5), np.uint16),
            photometric="minisblack",
            metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "CD3"]}},
        )
        assert metadata.ome is not None
        metadata.ome.images[0].pixels.channels.pop()

        with pytest.raises(ReaderFormatError, match="1 OME channels for 2 raster channels") as caught:
            normalize_cyx_image(raster, None, tiff=(metadata, series), context=_CONTEXT, path=_PATH)

        assert caught.value.context == _CONTEXT

    @pytest.mark.parametrize(
        ("shape", "axes", "found"),
        [
            ((2, 4, 5), "ZYX", "Axis 'z' has length 2 and is absent from target_axes."),
            ((2, 3, 4, 5), "CSYX", "source_axes contains duplicate axes: ('c', 'c', 'y', 'x')."),
            ((3, 4, 5), "CZX", "Raster source axes must contain 'x' and 'y'"),
        ],
    )
    def test_declared_axes_that_cannot_become_cyx_are_malformed_content(
        self, shape: tuple[int, ...], axes: str, found: str
    ) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            normalize_cyx_image(_raster(shape, axes), None, tiff=None, context=_CONTEXT, path=_PATH)

        assert caught.value.context == _CONTEXT
        assert caught.value.path == _PATH
        assert caught.value.found.startswith(found)
        assert caught.value.expected == _EXPECTED
        assert type(caught.value.__cause__) is ValueError

    @pytest.mark.parametrize(
        ("data_axes", "match"),
        [("yx", "source_axes rank 2 does not match array rank 3"), ("zyx", "Axis 'z' has length 2")],
    )
    def test_caller_axes_that_do_not_fit_are_a_caller_error_with_context(self, data_axes: str, match: str) -> None:
        with pytest.raises(ValueError, match=match) as caught:
            normalize_cyx_image(_raster((2, 4, 5), "CYX"), data_axes, tiff=None, context=_CONTEXT, path=_PATH)

        assert type(caught.value) is ValueError
        assert caught.value.__notes__ == [f"While handling {_CONTEXT.describe(path=_PATH)}."]
