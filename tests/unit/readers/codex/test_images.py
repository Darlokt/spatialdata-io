"""CODEX stitched-image options, axes, channels, pyramids and laziness."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from spatialdata.transformations import BaseTransformation, Identity, get_transformation
from xarray import DataArray, DataTree

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import inspect_tiff, make_raster_options
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.codex import codex
from spatialdata_io.readers.codex._images import read_image, validate_image_options
from spatialdata_io.readers.codex._layout import MarkerImage, MarkerSet, Montage
from tests.integration.readers.codex._fixtures import write_csv, write_image, write_uploader_montage

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pytest_mock import MockerFixture
    from spatialdata import SpatialData

DEFAULT_OPTIONS = make_raster_options()


class TestValidateImageOptions:
    """Reject invalid public options before expensive source work."""

    def test_valid(self) -> None:
        validate_image_options(None, None)
        validate_image_options("TZCYX", 0)
        validate_image_options("yxs", 3)

    def test_negative_series(self) -> None:
        with pytest.raises(ValueError, match="tiff_series must be nonnegative"):
            validate_image_options(None, -1)

    @pytest.mark.parametrize("series", [True, 1.5, "0"])
    def test_series_type(self, series: object) -> None:
        with pytest.raises(TypeError, match="tiff_series"):
            validate_image_options(None, series)  # pyrefly: ignore[bad-argument-type] - Untyped caller input.

    @pytest.mark.parametrize("axes", [["c", "y", "x"], ("C", "Y", "X"), b"CYX"])
    def test_axes_type(self, axes: object) -> None:
        with pytest.raises(
            TypeError, match=f"data_axes must be a string such as 'CYX', or None; found {type(axes).__name__}"
        ):
            validate_image_options(axes, None)  # pyrefly: ignore[bad-argument-type] - Untyped caller input.

    # Samples beside channels, or axes without X and Y, now fail before any I/O.
    @pytest.mark.parametrize("axes", ["", "C Y", "CY1", "CYy", "c-y-x", "CYXS", "CZT", "ÄYX"])
    def test_invalid_axes(self, axes: str) -> None:
        with pytest.raises(ValueError, match="distinct single-letter axes including X and Y"):
            validate_image_options(axes, None)

    def test_options_are_validated_before_io(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="data_axes"):
            codex(tmp_path / "missing", data_axes="CYY")

    @pytest.mark.parametrize(
        "read",
        [
            partial(codex, chunks=8),
            partial(codex, scale_factors=(2,)),
            partial(codex, data_axes="CYX"),
            partial(codex, tiff_series=0),
        ],
        ids=["chunks", "scale_factors", "data_axes", "tiff_series"],
    )
    def test_nondefault_options_require_an_image(self, tmp_path: Path, read: Callable[[Path], SpatialData]) -> None:
        write_csv(tmp_path / "cells.csv")

        with pytest.raises(ValueError, match="require an image"):
            read(tmp_path)

    def test_default_raster_options_need_no_image(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "cells.csv")

        result = codex(tmp_path, scale_factors=(), chunks=(1, 1024, 1024))

        assert dict(result.images) == {}

    def test_invalid_raster_option_fails_before_discovery(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="scale_factors"):
            codex(tmp_path / "missing", scale_factors=(1,))


class TestReadImage:
    """Keep explicit channel semantics, TIFF axes, and small images lazy."""

    @pytest.mark.parametrize(("dtype", "shape"), [("uint16", (2, 16, 20)), ("uint8", (16, 20)), ("uint16", (18, 20))])
    def test_marker_planes_have_one_plane_and_a_common_shape_and_dtype(
        self, tmp_path: Path, dtype: str, shape: tuple[int, ...]
    ) -> None:
        first = ArtifactFile(tmp_path / "first.tif")
        second = ArtifactFile(tmp_path / "second.tif")
        tifffile.imwrite(first, np.zeros((16, 20), dtype=np.uint16))
        tifffile.imwrite(
            second,
            np.zeros(shape, dtype=dtype),
            photometric="minisblack",
            metadata={
                "axes": "CYX" if len(shape) == 3 else "YX",
            },
        )

        with pytest.raises(ReaderFormatError, match="one plane per marker, all sharing"):
            read_image(
                MarkerSet((MarkerImage(first, 1, 1, "A"), MarkerImage(second, 1, 2, "B"))),
                data_axes=None,
                tiff_series=None,
                raster=DEFAULT_OPTIONS,
                frame="image_pixels",
                transformation=Identity(),
            )

    @pytest.mark.parametrize("labels", [None, ["A", "", "C", "D", "E", "F"]])
    def test_montage_without_complete_names_uses_acquisition_indices(
        self, tmp_path: Path, labels: list[str] | None
    ) -> None:
        path, pixels = write_uploader_montage(tmp_path, "reg001", labels=labels)
        image = read_image(
            Montage(ArtifactFile(path), None),
            data_axes=None,
            tiff_series=None,
            raster=DEFAULT_OPTIONS,
            frame="image_pixels",
            transformation=Identity(),
        )
        assert isinstance(image, DataArray)
        assert image.c.values.tolist() == [
            "cyc001_ch001",
            "cyc001_ch002",
            "cyc001_ch003",
            "cyc002_ch001",
            "cyc002_ch002",
            "cyc002_ch003",
        ]
        np.testing.assert_array_equal(image.data.compute(), pixels)

    @pytest.mark.parametrize(
        ("data", "match"),
        [
            pytest.param(b"A\n", "6 nonblank channel names", id="incorrect-count"),
            pytest.param(b"\xff", "UTF-8 or BOM-marked", id="invalid-encoding"),
            pytest.param(b"x" * (1024**2 + 1), "at most 1 MiB", id="bounded-read"),
        ],
    )
    def test_invalid_montage_sidecar(self, tmp_path: Path, data: bytes, match: str) -> None:
        path, _ = write_uploader_montage(tmp_path, "reg001")
        sidecar = ArtifactFile(path.parent / "channelNames.txt")
        sidecar.write_bytes(data)
        with pytest.raises(ReaderFormatError, match=match):
            read_image(
                Montage(ArtifactFile(path), sidecar),
                data_axes=None,
                tiff_series=None,
                raster=DEFAULT_OPTIONS,
                frame="image_pixels",
                transformation=Identity(),
            )

    def test_montage_rejects_nonsingleton_z_axis(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "reg001_montage.tif")
        tifffile.imwrite(path, np.zeros((3, 16, 20), dtype=np.uint16), imagej=True, metadata={"axes": "ZYX"})
        source = Montage(path, None)
        with pytest.raises(ReaderFormatError, match="an Uploader montage with declared cycle"):
            read_image(
                source,
                data_axes=None,
                tiff_series=None,
                raster=DEFAULT_OPTIONS,
                frame="image_pixels",
                transformation=Identity(),
            )
        with pytest.raises(ValueError, match="rank") as caught:
            read_image(
                source,
                data_axes="YX",
                tiff_series=None,
                raster=DEFAULT_OPTIONS,
                frame="image_pixels",
                transformation=Identity(),
            )
        assert any("reg001_montage.tif" in note for note in caught.value.__notes__)

    def test_montage_names_follow_explicit_axis_reinterpretation(self, tmp_path: Path) -> None:
        labels = ["A", "B", "C", "D", "E", "F"]
        path, source_pixels = write_uploader_montage(tmp_path, "reg001", labels=labels)
        image = read_image(
            Montage(ArtifactFile(path), None),
            data_axes="CTYX",
            tiff_series=None,
            raster=DEFAULT_OPTIONS,
            frame="image_pixels",
            transformation=Identity(),
        )
        assert isinstance(image, DataArray)
        assert image.c.values.tolist() == ["A", "D", "B", "E", "C", "F"]
        np.testing.assert_array_equal(image.data.compute(), source_pixels[[0, 3, 1, 4, 2, 5]])

    def test_ome_independent_xy_calibration(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "image.ome.tif")
        tifffile.imwrite(
            path,
            np.zeros((16, 20), dtype=np.uint16),
            ome=True,
            metadata={
                "PhysicalSizeX": 800,
                "PhysicalSizeXUnit": "nm",
                "PhysicalSizeY": 0.4,
                "PhysicalSizeYUnit": "µm",
            },
        )
        image = read_image(
            path,
            data_axes=None,
            tiff_series=None,
            raster=DEFAULT_OPTIONS,
            frame="image_pixels",
            transformation=Identity(),
            physical_frame="global",
        )
        transform = get_transformation(image, to_coordinate_system="global")
        assert isinstance(transform, BaseTransformation)
        np.testing.assert_allclose(transform.to_affine_matrix(("x", "y"), ("x", "y")), np.diag([0.8, 0.4, 1]))

        swapped = read_image(
            path,
            data_axes="XY",
            tiff_series=None,
            raster=DEFAULT_OPTIONS,
            frame="image_pixels",
            transformation=Identity(),
            physical_frame="global",
        )
        assert isinstance(get_transformation(swapped, to_coordinate_system="image_pixels"), Identity)

    def test_malformed_tiff_has_reader_context(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        path.write_bytes(b"invalid TIFF")

        with pytest.raises(ValueError, match="TIFF") as caught:
            read_image(
                ArtifactFile(path),
                data_axes=None,
                tiff_series=None,
                raster=DEFAULT_OPTIONS,
                frame="image_pixels",
                transformation=Identity(),
            )

        assert any("codex" in note and "stitched image" in note for note in caught.value.__notes__)

    def test_large_image_is_not_pyramided_automatically(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        image = write_image(path)

        result = read_image(
            ArtifactFile(path),
            data_axes=None,
            tiff_series=None,
            raster=DEFAULT_OPTIONS,
            frame="image_pixels",
            transformation=Identity(),
        )

        assert isinstance(result, DataArray)
        # Default tiles exceed the image, so each chunk is one whole channel.
        assert result.data.chunksize == (1, 32, 40)
        np.testing.assert_array_equal(result.data.compute(), image)

    def test_scale_factors_build_a_pyramid(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        image = write_image(path)

        result = read_image(
            ArtifactFile(path),
            data_axes=None,
            tiff_series=None,
            raster=make_raster_options((2, 2), (1, 8, 8)),
            frame="image_pixels",
            transformation=Identity(),
        )

        assert isinstance(result, DataTree)
        assert list(result.children) == ["scale0", "scale1", "scale2"]
        scale0 = result["scale0"]["image"]
        assert scale0.data.chunksize == (1, 8, 8)
        np.testing.assert_array_equal(scale0.data.compute(), image)
        assert result["scale1"]["image"].shape == (2, 16, 20)
        assert result["scale2"]["image"].shape == (2, 8, 10)

    def test_small_grayscale_is_single_scale(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        tifffile.imwrite(path, np.ones((2, 3), dtype=np.uint8), metadata={"axes": "YX"})

        result = read_image(
            ArtifactFile(path),
            data_axes=None,
            tiff_series=None,
            raster=DEFAULT_OPTIONS,
            frame="image_pixels",
            transformation=Identity(),
        )

        assert isinstance(result, DataArray)
        assert result.shape == (1, 2, 3)
        np.testing.assert_array_equal(result.data.compute(), np.ones((1, 2, 3), dtype=np.uint8))

    @pytest.mark.parametrize("channels", [3, 4, 5])
    def test_arbitrary_channels_are_not_rgb(self, tmp_path: Path, channels: int) -> None:
        path = tmp_path / "image.tif"
        tifffile.imwrite(
            path, np.zeros((channels, 8, 10), dtype=np.uint16), metadata={"axes": "CYX"}, photometric="minisblack"
        )

        result = read_image(
            ArtifactFile(path),
            data_axes=None,
            tiff_series=None,
            raster=DEFAULT_OPTIONS,
            frame="image_pixels",
            transformation=Identity(),
        )

        assert result.coords["c"].to_numpy().tolist() == list(range(channels))

    def test_rgb_samples_are_ordered_channels(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        tifffile.imwrite(path, image, photometric="rgb")

        result = read_image(
            ArtifactFile(path),
            data_axes=None,
            tiff_series=None,
            raster=DEFAULT_OPTIONS,
            frame="image_pixels",
            transformation=Identity(),
        )

        np.testing.assert_array_equal(result.data.compute(), image.transpose(2, 0, 1))

    def test_unknown_stack_axes_need_override(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        tifffile.imwrite(
            path, np.zeros((2, 8, 10), dtype=np.uint16), metadata={"axes": "QYX"}, photometric="minisblack"
        )

        with pytest.raises(ReaderFormatError, match="Axis 'q'") as caught:
            read_image(
                ArtifactFile(path),
                data_axes=None,
                tiff_series=None,
                raster=DEFAULT_OPTIONS,
                frame="image_pixels",
                transformation=Identity(),
            )
        assert caught.value.context == ReaderErrorContext("codex", "stitched image")
        assert caught.value.path == path
        assert type(caught.value.__cause__) is ValueError

        result = read_image(
            ArtifactFile(path),
            data_axes="CYX",
            tiff_series=None,
            raster=DEFAULT_OPTIONS,
            frame="image_pixels",
            transformation=Identity(),
        )

        assert result.shape == (2, 8, 10)

    @pytest.mark.parametrize(
        ("names", "expected"),
        [
            (["CD3", "DAPI"], ["CD3", "DAPI"]),
            (["CD3", "CD3"], ["Channel:0:0", "Channel:0:1"]),
            (["CD3", ""], ["Channel:0:0", "Channel:0:1"]),
            (["Channel:0:1", ""], ["Channel:0:0", "Channel:0:1"]),
        ],
    )
    def test_ome_channel_identities_are_collectively_unique(
        self, tmp_path: Path, names: list[str], expected: list[str]
    ) -> None:
        path = tmp_path / "image.ome.tif"
        source = np.arange(2 * 8 * 10, dtype=np.uint16).reshape(2, 8, 10)
        tifffile.imwrite(
            path, source, ome=True, photometric="minisblack", metadata={"axes": "CYX", "Channel": {"Name": names}}
        )
        result = read_image(
            ArtifactFile(path),
            data_axes=None,
            tiff_series=None,
            raster=DEFAULT_OPTIONS,
            frame="image_pixels",
            transformation=Identity(),
        )
        assert result.coords["c"].to_numpy().tolist() == expected
        np.testing.assert_array_equal(result.data.compute(), source)

    def test_ome_channel_count_mismatch_is_rejected(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "image.ome.tif"
        tifffile.imwrite(
            path,
            np.zeros((2, 8, 10), dtype=np.uint16),
            ome=True,
            photometric="minisblack",
            metadata={"axes": "CYX", "Channel": {"Name": ["CD3", "DAPI"]}},
        )
        metadata = inspect_tiff(ArtifactFile(path), context=ReaderErrorContext("test", "raster"))
        assert metadata.ome is not None
        metadata.ome.images[0].pixels.channels.pop()
        mocker.patch("spatialdata_io.readers.codex._images.inspect_tiff", return_value=metadata)

        with pytest.raises(ReaderFormatError, match="1 OME channels for 2 raster channels"):
            read_image(
                ArtifactFile(path),
                data_axes=None,
                tiff_series=None,
                raster=DEFAULT_OPTIONS,
                frame="image_pixels",
                transformation=Identity(),
            )

    def test_axes_rank_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        write_image(path)

        with pytest.raises(ValueError, match="rank 2 does not match array rank 3") as caught:
            read_image(
                ArtifactFile(path),
                data_axes="YX",
                tiff_series=None,
                raster=DEFAULT_OPTIONS,
                frame="image_pixels",
                transformation=Identity(),
            )

        # Caller axes that do not fit the file are a caller error, not malformed content.
        assert type(caught.value) is ValueError
        context = ReaderErrorContext("codex", "stitched image")
        assert caught.value.__notes__ == [f"While handling {context.describe(path=path)}."]

    def test_axis_reinterpreted_as_channels_gets_positions(self, tmp_path: Path) -> None:
        path = tmp_path / "image.ome.tif"
        source = np.arange(3 * 8 * 10, dtype=np.uint16).reshape(3, 8, 10)
        # One OME channel record describes the single channel, not the three Z planes.
        tifffile.imwrite(
            path, source, ome=True, photometric="minisblack", metadata={"axes": "ZYX", "Channel": {"Name": ["DAPI"]}}
        )

        result = read_image(
            ArtifactFile(path),
            data_axes="CYX",
            tiff_series=None,
            raster=DEFAULT_OPTIONS,
            frame="image_pixels",
            transformation=Identity(),
        )

        assert result.coords["c"].to_numpy().tolist() == [0, 1, 2]
        np.testing.assert_array_equal(result.data.compute(), source)

    @pytest.mark.parametrize(
        ("ome", "samples", "expected"),
        [
            pytest.param(True, (3, "rgb", None), ["r", "g", "b"], id="ome-rgb"),
            pytest.param(True, (4, "rgb", ["unassalpha"]), ["r", "g", "b", "a"], id="ome-rgba"),
            pytest.param(
                True,
                (3, "minisblack", None),
                [f"Channel:0:0:sample:{index}" for index in range(3)],
                id="ome-minisblack",
            ),
            pytest.param(False, (3, "rgb", None), ["r", "g", "b"], id="plain-rgb"),
            pytest.param(False, (3, "minisblack", None), [0, 1, 2], id="plain-minisblack"),
        ],
    )
    def test_sample_channels_follow_the_declared_colour_model(
        self,
        tmp_path: Path,
        samples: tuple[int, str, list[str] | None],
        expected: list[str | int],
        *,
        ome: bool,
    ) -> None:
        """Declared RGB(A) samples are r, g, b(, a); other OME samples are named by record and sample."""
        count, photometric, extrasamples = samples
        path = tmp_path / "image.tif"
        image = np.arange(8 * 10 * count, dtype=np.uint8).reshape(8, 10, count)
        tifffile.imwrite(
            path, image, ome=ome, photometric=photometric, planarconfig="contig", extrasamples=extrasamples
        )

        result = read_image(
            ArtifactFile(path),
            data_axes=None,
            tiff_series=None,
            raster=DEFAULT_OPTIONS,
            frame="image_pixels",
            transformation=Identity(),
        )

        assert result.coords["c"].to_numpy().tolist() == expected
        np.testing.assert_array_equal(result.data.compute(), image.transpose(2, 0, 1))

    def test_scale_factors_exceeding_the_extent(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        write_image(path)

        with pytest.raises(ValueError, match=r"exceed the spatial extent of the stitched image \(32 x 40 pixels\)"):
            read_image(
                ArtifactFile(path),
                data_axes=None,
                tiff_series=None,
                raster=make_raster_options((4, 4, 4)),
                frame="image_pixels",
                transformation=Identity(),
            )

    def test_series_selection_out_of_range(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        write_image(path)

        with pytest.raises(IndexError, match="series index 1") as caught:
            read_image(
                ArtifactFile(path),
                data_axes=None,
                tiff_series=1,
                raster=DEFAULT_OPTIONS,
                frame="image_pixels",
                transformation=Identity(),
            )

        context = ReaderErrorContext("codex", "stitched image")
        assert caught.value.__notes__ == [f"While handling {context.describe(path=path)}."]
